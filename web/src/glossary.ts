/** Short explanations of attention names, with links to the papers. */
export const GLOSSARY: Record<string, { text: string; url: string }> = {
  MHA: { text: "every query head has its own key and value head.", url: "https://arxiv.org/abs/1706.03762" },
  MQA: { text: "all query heads share a single key/value head.", url: "https://arxiv.org/abs/1911.02150" },
  GQA: {
    text: "query heads share key/value heads in groups; the KV cache shrinks by heads / KV heads.",
    url: "https://arxiv.org/abs/2305.13245",
  },
  MLA: {
    text: "keys and values are expanded from a shared low-rank latent, which is all that is cached, plus a small decoupled RoPE key.",
    url: "https://arxiv.org/abs/2405.04434",
  },
  SWA: {
    text: "each token attends only to the last W tokens, so the layer's KV cache is bounded by the window.",
    url: "https://arxiv.org/abs/2004.05150",
  },
  DSA: {
    text: "DeepSeek Sparse Attention: a light indexer scores past tokens and each query attends only to the top-k.",
    url: "https://arxiv.org/abs/2512.02556",
  },
  CSA: {
    text: "Compressed Sparse Attention: compresses past KV entries along the sequence, then selects entries with a sparse-attention indexer.",
    url: "https://arxiv.org/abs/2606.19348",
  },
  CSA2: {
    text: "Compressed Sparse Attention 2: adds cross-layer reuse of compressed KV caches to reduce storage further.",
    url: "https://arxiv.org/abs/2609.19969",
  },
  HCA: {
    text: "Heavily Compressed Attention: compresses the KV sequence more aggressively and attends densely to the compressed entries.",
    url: "https://arxiv.org/abs/2606.19348",
  },
  "Gated DeltaNet": {
    text: "linear attention with a fixed-size state updated by a gated delta rule; no KV cache that grows with context.",
    url: "https://arxiv.org/abs/2412.06464",
  },
  KDA: {
    text: "Kimi Delta Attention: Gated DeltaNet with channel-wise gating; fixed-size state.",
    url: "https://arxiv.org/abs/2510.26692",
  },
  Lightning: {
    text: "Lightning Attention: MiniMax's block-wise linear attention; fixed-size state.",
    url: "https://arxiv.org/abs/2501.08313",
  },
  "Mamba-2": {
    text: "selective state-space layer (structured state-space duality); fixed-size state.",
    url: "https://arxiv.org/abs/2405.21060",
  },
};

/** Glossary terms named in a token-mixer description, in glossary order. */
export function terms(kind: string): string[] {
  return Object.keys(GLOSSARY).filter((t) => new RegExp(`(^|[^\\w-])${t}($|[^\\w-])`).test(kind));
}
