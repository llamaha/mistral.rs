//! DSpine code bridge (<https://arxiv.org/abs/2609.36173>, reference <https://github.com/Tencent/DSpine>).
//!
//! DSpine drafters are DFlash layers whose residual stream gets a gated write from each block
//! row's predecessor after every layer, so causal conditioning flows through depth while all rows
//! update in parallel. Proposals then come from a serial refinement over the top-k candidates of
//! each row that re-scores the last write against the predecessor token actually chosen.

use std::sync::Arc;

use candle_core::{DType, IndexOp, Result, Tensor, D};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use serde::Deserialize;

use crate::layers::RmsNorm;

// torch.finfo(torch.float32).tiny, the reference's clamp floor for the read RMS.
const F32_TINY: f64 = 1.175_494_350_822_287_5e-38;

#[derive(Debug, Clone, Deserialize)]
pub struct DSpineBridgeConfig {
    pub code_size: usize,
    pub rank: usize,
    pub eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DSpineRefineConfig {
    pub head_topk: usize,
}

pub(crate) struct DSpineLoadConfig<'a> {
    pub bridge: &'a DSpineBridgeConfig,
    pub feature_layers: Option<&'a [usize]>,
    pub head_topk: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

/// Last write's pre-write state for the proposal rows (block rows 1..).
pub(crate) struct DSpineStash {
    // [b, n, hidden] f32
    raw: Tensor,
    // [b, n, 1] f32
    rms: Tensor,
    // [b, n, rank] f32
    gate_query: Tensor,
}

pub(crate) struct DSpineBridge {
    // per draft layer, [code, hidden]; the last layer reads through `read_weight`
    read_weights: Vec<Tensor>,
    // per draft layer, [hidden, rank]
    write_weights: Vec<Tensor>,
    // [3 * rank, code] = [gate; predecessor_gate; prev]
    packed_weight: Tensor,
    // [rank] f32
    gate_bias: Tensor,
    // [vocab, 2 * rank]: sqrt(code) * codebook @ [predecessor_gate; prev]^T
    sender_table: Tensor,
    // [vocab, rank]: (lm_head * final norm) @ last write, filled by `prepare`
    candidate_gains: Option<Tensor>,
    rank: usize,
    code_size: usize,
    eps: f64,
    top_k: usize,
}

impl DSpineBridge {
    pub(crate) fn load(vb: ShardedVarBuilder, cfg: DSpineLoadConfig<'_>) -> Result<Self> {
        let DSpineLoadConfig {
            bridge,
            feature_layers,
            head_topk,
            num_layers,
            hidden_size,
            vocab_size,
        } = cfg;
        // Config ids are 1-based; the bridge writes after every layer in all released checkpoints.
        let expected = (1..=num_layers).collect::<Vec<_>>();
        if feature_layers.is_some_and(|layers| layers != expected.as_slice()) {
            candle_core::bail!(
                "DSpine drafts must write after every draft layer; got selected_prefix_feature_layers {feature_layers:?}"
            );
        }
        if num_layers < 2 {
            candle_core::bail!("DSpine drafts need at least two layers");
        }
        let (c, r, h) = (bridge.code_size, bridge.rank, hidden_size);
        let early = vb.get((num_layers - 1, c, h), "early_read_weights")?;
        let mut read_weights = (0..num_layers - 1)
            .map(|i| early.i(i)?.contiguous())
            .collect::<Result<Vec<_>>>()?;
        read_weights.push(vb.get((c, h), "read_weight")?);
        let writes = vb.get((num_layers, h, r), "feature_write_weights")?;
        let write_weights = (0..num_layers)
            .map(|i| writes.i(i)?.contiguous())
            .collect::<Result<Vec<_>>>()?;
        let gate = vb.pp("gate").get((r, c), "weight")?;
        let predecessor_gate = vb.pp("predecessor_gate").get((r, c), "weight")?;
        let prev = vb.pp("prev").get((r, c), "weight")?;
        let packed_weight = Tensor::cat(&[&gate, &predecessor_gate, &prev], 0)?;
        let gate_bias = vb.pp("gate").get(r, "bias")?.to_dtype(DType::F32)?;

        // The codebook only ever enters through these two projections, so drop it once folded.
        #[allow(clippy::cast_precision_loss)]
        let sqrt_c = (c as f64).sqrt();
        let codebook = vb.get((vocab_size, c), "codebook")?;
        let sender_weight = Tensor::cat(&[&predecessor_gate, &prev], 0)?;
        let sender_table = (codebook.to_dtype(DType::F32)? * sqrt_c)?
            .to_dtype(codebook.dtype())?
            .matmul(&sender_weight.t()?)?
            .contiguous()?;
        Ok(Self {
            read_weights,
            write_weights,
            packed_weight,
            gate_bias,
            sender_table,
            candidate_gains: None,
            rank: r,
            code_size: c,
            eps: bridge.eps,
            top_k: head_topk,
        })
    }

    /// Folds the target head and the drafter's final norm into the last write's per-candidate gains.
    pub(crate) fn prepare(
        &mut self,
        lm_head: &Arc<dyn QuantMethod>,
        norm_weight: &Tensor,
    ) -> Result<()> {
        let last = self
            .write_weights
            .last()
            .expect("load checks the layer count");
        let scaled = last
            .to_dtype(DType::F32)?
            .broadcast_mul(&norm_weight.to_dtype(DType::F32)?.unsqueeze(1)?)?
            .t()?
            .to_dtype(last.dtype())?
            .unsqueeze(0)?
            .contiguous()?;
        // [1, rank, vocab] -> [vocab, rank]
        let gains = lm_head.forward(&scaled)?.squeeze(0)?.t()?.contiguous()?;
        self.candidate_gains = Some(gains);
        Ok(())
    }

    pub(crate) fn top_k(&self) -> usize {
        self.top_k
    }

    /// `[b, 1, 2 * rank]` f32 sender rows for each block's anchor token.
    pub(crate) fn anchor_senders(&self, anchors: &Tensor) -> Result<Tensor> {
        let b = anchors.dim(0)?;
        self.sender_table
            .index_select(anchors, 0)?
            .to_dtype(DType::F32)?
            .reshape((b, 1, 2 * self.rank))
    }

    /// One write after `layer` on the residual stream `hs` `[b, block, hidden]`.
    pub(crate) fn write(
        &self,
        hs: &Tensor,
        layer: usize,
        anchor_senders: &Tensor,
        stash: bool,
    ) -> Result<(Tensor, Option<DSpineStash>)> {
        let (_, block, _) = hs.dims3()?;
        let n = block - 1;
        let r = self.rank;
        let dtype = hs.dtype();
        #[allow(clippy::cast_precision_loss)]
        let sqrt_c = (self.code_size as f64).sqrt();
        let f = hs.to_dtype(DType::F32)?;
        let raw = f.narrow(1, 1, n)?;
        let rms = raw
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .maximum(F32_TINY)?
            .sqrt()?;
        let normalized = raw.broadcast_div(&(&rms + self.eps)?)?.to_dtype(dtype)?;
        let projected = normalized
            .broadcast_matmul(&self.read_weights[layer].t()?)?
            .to_dtype(DType::F32)?;
        let projected_norm = projected.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let codes =
            (projected.broadcast_div(&(projected_norm + self.eps)?)? * sqrt_c)?.to_dtype(dtype)?;
        let packed = codes
            .broadcast_matmul(&self.packed_weight.t()?)?
            .to_dtype(DType::F32)?;
        let gate_query = packed.narrow(D::Minus1, 0, r)?.contiguous()?;
        // Row t is written from row t - 1; the anchor's sender comes from its true token.
        let senders = if n > 1 {
            Tensor::cat(
                &[
                    anchor_senders,
                    &packed.narrow(1, 0, n - 1)?.narrow(D::Minus1, r, 2 * r)?,
                ],
                1,
            )?
        } else {
            anchor_senders.clone()
        };
        let gate = candle_nn::ops::sigmoid(
            &(&gate_query + senders.narrow(D::Minus1, 0, r)?)?.broadcast_add(&self.gate_bias)?,
        )?;
        let message = (gate * senders.narrow(D::Minus1, r, r)?)?.to_dtype(dtype)?;
        let delta = message
            .broadcast_matmul(&self.write_weights[layer].t()?)?
            .to_dtype(DType::F32)?;
        let written = (&raw + delta.broadcast_mul(&rms)?)?;
        let out = Tensor::cat(&[&f.narrow(1, 0, 1)?, &written], 1)?.to_dtype(dtype)?;
        let stash = stash.then_some(DSpineStash {
            raw,
            rms,
            gate_query,
        });
        Ok((out, stash))
    }

    /// Greedy serial refinement of the final-normed rows 1.. `[b, n, hidden]` into `[b, n]` u32 drafts.
    pub(crate) fn refine_greedy(&self, inputs: DSpineRefineInputs<'_>) -> Result<Tensor> {
        let table = self.candidate_table(inputs, false)?;
        let best = table.scores.argmax(D::Minus1)?;
        let n = best.dim(1)?;
        // Row 0's senders are all the anchor, so any lane starts the walk.
        let mut lane = best.i((.., 0, 0..1))?.contiguous()?;
        let mut tokens = Vec::with_capacity(n);
        for i in 0..n {
            tokens.push(
                table
                    .candidates
                    .i((.., i, ..))?
                    .contiguous()?
                    .gather(&lane, 1)?,
            );
            if i + 1 < n {
                lane = best.i((.., i + 1, ..))?.contiguous()?.gather(&lane, 1)?;
            }
        }
        Tensor::cat(&tokens, 1)?.contiguous()
    }

    /// Sampled serial chain: `(tokens [b, n], candidates [b, n, k], probs [b, n, k])`. Rows with a
    /// non-positive inverse temperature draft greedily with one-hot probabilities.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn refine_sampled(
        &self,
        inputs: DSpineRefineInputs<'_>,
        inverse_temperatures: &Tensor,
        uniforms: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let table = self.candidate_table(inputs, true)?;
        let (b, n, k, _) = table.scores.dims4()?;
        let inverse_temperatures = inverse_temperatures.reshape((b, 1))?;
        let greedy_rows = inverse_temperatures
            .le(0.0)?
            .broadcast_as((b, k))?
            .contiguous()?;
        let mut lane = Tensor::zeros((b, 1), DType::U32, table.scores.device())?;
        let mut tokens = Vec::with_capacity(n);
        let mut probs = Vec::with_capacity(n);
        for i in 0..n {
            let rows = table.scores.i((.., i))?.contiguous()?;
            let pick = lane
                .reshape((b, 1, 1))?
                .broadcast_as((b, 1, k))?
                .contiguous()?;
            let row = rows.gather(&pick, 1)?.squeeze(1)?;
            let max = row.max_keepdim(D::Minus1)?;
            let weights = row
                .broadcast_sub(&max)?
                .broadcast_mul(&inverse_temperatures)?
                .exp()?;
            let sampled = weights.broadcast_div(&weights.sum_keepdim(D::Minus1)?)?;
            let one_hot = row.broadcast_eq(&max)?.to_dtype(DType::F32)?;
            let one_hot = one_hot.broadcast_div(&one_hot.sum_keepdim(D::Minus1)?)?;
            let q = greedy_rows.where_cond(&one_hot, &sampled)?;
            // First slot whose cumulative mass passes the draw, as in the reference sampler.
            let below = q
                .cumsum(D::Minus1)?
                .broadcast_le(&uniforms.i((.., i..i + 1))?)?
                .to_dtype(DType::F32)?
                .sum_keepdim(D::Minus1)?;
            #[allow(clippy::cast_precision_loss)]
            let sampled_slot = below.minimum((k - 1) as f64)?.to_dtype(DType::U32)?;
            let greedy_slot = row.argmax_keepdim(D::Minus1)?;
            lane = greedy_rows
                .i((.., 0..1))?
                .contiguous()?
                .where_cond(&greedy_slot, &sampled_slot)?;
            tokens.push(
                table
                    .candidates
                    .i((.., i, ..))?
                    .contiguous()?
                    .gather(&lane, 1)?,
            );
            probs.push(q.unsqueeze(1)?);
        }
        Ok((
            Tensor::cat(&tokens, 1)?.contiguous()?,
            table.candidates.contiguous()?,
            Tensor::cat(&probs, 1)?.contiguous()?,
        ))
    }

    /// Per row, sender lane by candidate lane scores; `normalized` applies the final RMS norm of
    /// each refined row so the scores are true logits.
    fn candidate_table(
        &self,
        inputs: DSpineRefineInputs<'_>,
        normalized: bool,
    ) -> Result<CandidateTable> {
        let DSpineRefineInputs {
            hidden,
            stash,
            anchors,
            norm,
            lm_head,
        } = inputs;
        let gains_table = self.candidate_gains.as_ref().ok_or_else(|| {
            candle_core::Error::msg(
                "DSpine candidate gains were not prepared against the target head",
            )
        })?;
        let (b, n, h) = hidden.dims3()?;
        let (k, r) = (self.top_k, self.rank);
        let norm_weight = norm.weight().to_dtype(DType::F32)?;
        let scaled_raw = stash
            .raw
            .broadcast_mul(&norm_weight)?
            .to_dtype(hidden.dtype())?;
        // One head pass scores both the drafted rows and the pre-write rows the refinement re-adds to.
        let logits = lm_head.forward(&Tensor::cat(&[hidden, &scaled_raw], 1)?)?;
        let vocab = logits.dim(D::Minus1)?;
        let draft_logits = logits
            .narrow(1, 0, n)?
            .reshape((b * n, vocab))?
            .contiguous()?;
        let (candidates, _topk) = top_k_ids(&draft_logits, k)?;
        let base = logits
            .narrow(1, n, n)?
            .reshape((b * n, vocab))?
            .contiguous()?
            .gather(&candidates, 1)?
            .to_dtype(DType::F32)?
            .reshape((b, n, 1, k))?;
        let candidates = candidates.reshape((b, n, k))?;
        let anchor_rows = anchors.reshape((b, 1, 1))?.broadcast_as((b, 1, k))?;
        let senders = if n > 1 {
            Tensor::cat(&[&anchor_rows, &candidates.narrow(1, 0, n - 1)?], 1)?
        } else {
            anchor_rows.contiguous()?
        };
        let sender_rows = self
            .sender_table
            .index_select(&senders.flatten_all()?, 0)?
            .to_dtype(DType::F32)?
            .reshape((b, n, k, 2 * r))?;
        let gate = candle_nn::ops::sigmoid(
            &sender_rows
                .narrow(D::Minus1, 0, r)?
                .broadcast_add(&stash.gate_query.unsqueeze(2)?)?
                .broadcast_add(&self.gate_bias)?,
        )?;
        let message = (gate * sender_rows.narrow(D::Minus1, r, r)?)?.reshape((b * n, k, r))?;
        let gains = gains_table
            .index_select(&candidates.flatten_all()?, 0)?
            .to_dtype(DType::F32)?
            .reshape((b * n, k, r))?;
        let pair = message
            .matmul(&gains.t()?.contiguous()?)?
            .reshape((b, n, k, k))?;
        let rms = stash.rms.unsqueeze(3)?;
        let mut scores = pair.broadcast_mul(&rms)?.broadcast_add(&base)?;
        if normalized {
            let last = self
                .write_weights
                .last()
                .expect("load checks the layer count");
            let delta = message
                .to_dtype(last.dtype())?
                .broadcast_matmul(&last.t()?)?
                .to_dtype(DType::F32)?
                .reshape((b, n, k, h))?;
            let refined = delta
                .broadcast_mul(&rms)?
                .broadcast_add(&stash.raw.unsqueeze(2)?)?;
            let inv_rms = (refined.sqr()?.mean_keepdim(D::Minus1)? + norm.eps())?
                .sqrt()?
                .recip()?
                .reshape((b, n, k, 1))?;
            scores = scores.broadcast_mul(&inv_rms)?;
        }
        Ok(CandidateTable { candidates, scores })
    }
}

pub(crate) struct DSpineRefineInputs<'a> {
    pub hidden: &'a Tensor,
    pub stash: &'a DSpineStash,
    pub anchors: &'a Tensor,
    pub norm: &'a RmsNorm,
    pub lm_head: &'a Arc<dyn QuantMethod>,
}

struct CandidateTable {
    // [b, n, k] u32
    candidates: Tensor,
    // [b, n, sender lane, candidate lane] f32
    scores: Tensor,
}

#[cfg(feature = "cuda")]
type TopKKeepalive = Option<crate::ops::RankedTopKPackedOutput>;
#[cfg(not(feature = "cuda"))]
type TopKKeepalive = Option<()>;

/// `[rows, k]` u32 top-k ids; the second value keeps the kernel workspace alive while the ids are used.
fn top_k_ids(logits: &Tensor, k: usize) -> Result<(Tensor, TopKKeepalive)> {
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() {
        let topk = crate::ops::cuda_topk_ranked_packed_batched(logits, k)?;
        let ids = topk
            .packed
            .narrow(1, k, k)?
            .to_dtype(DType::U32)?
            .contiguous()?;
        return Ok((ids, Some(topk)));
    }
    let ids = logits
        .to_dtype(DType::F32)?
        .arg_sort_last_dim(false)?
        .narrow(1, 0, k)?
        .contiguous()?;
    Ok((ids, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn bridge(device: &Device) -> Result<DSpineBridge> {
        let (layers, h, c, r, v) = (2, 8, 4, 3, 10);
        let rand = |shape: &[usize]| Tensor::randn(0f32, 1f32, shape, device);
        let codebook = rand(&[v, c])?;
        let pred = rand(&[r, c])?;
        let prev = rand(&[r, c])?;
        let sender_table = (codebook * 2.0)?.matmul(&Tensor::cat(&[&pred, &prev], 0)?.t()?)?;
        Ok(DSpineBridge {
            read_weights: (0..layers).map(|_| rand(&[c, h])).collect::<Result<_>>()?,
            write_weights: (0..layers).map(|_| rand(&[h, r])).collect::<Result<_>>()?,
            packed_weight: Tensor::cat(&[&rand(&[r, c])?, &pred, &prev], 0)?,
            gate_bias: rand(&[r])?,
            sender_table,
            candidate_gains: None,
            rank: r,
            code_size: c,
            eps: 1e-6,
            top_k: 4,
        })
    }

    #[test]
    fn anchor_row_is_untouched_and_each_row_reads_only_its_predecessor() -> Result<()> {
        let device = Device::Cpu;
        let bridge = bridge(&device)?;
        let anchors = Tensor::new(&[3u32, 7], &device)?;
        let senders = bridge.anchor_senders(&anchors)?;
        let hs = Tensor::randn(0f32, 1f32, (2, 5, 8), &device)?;
        let (out, stash) = bridge.write(&hs, 0, &senders, true)?;
        assert!(stash.is_some());
        let diff = (out.i((.., 0, ..))? - hs.i((.., 0, ..))?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(diff, 0.0);

        // Perturbing row 3 may only move rows 3 and 4.
        let bumped = hs.slice_assign(&[0..2, 3..4, 0..8], &(hs.i((.., 3..4, ..))? + 5.0)?)?;
        let (out2, _) = bridge.write(&bumped, 0, &senders, false)?;
        let moved = (out2 - &out)?.abs()?.sum(D::Minus1)?.to_vec2::<f32>()?;
        for row in moved {
            assert!(row[..3].iter().all(|d| *d == 0.0));
            assert!(row[3] > 0.0 && row[4] > 0.0);
        }
        Ok(())
    }
}
