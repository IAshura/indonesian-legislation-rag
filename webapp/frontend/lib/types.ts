export type TraceNodeType =
  | "user_ask" | "llm_ask" | "embed_query" | "retrieve_dense" | "retrieve_bm25"
  | "fuse_rrf" | "rerank" | "join_parent" | "build_prompt"
  | "llm_answer" | "tool_call";

export interface TraceNode {
  id: string;
  type: TraceNodeType;
  label: string;
  detail: Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
}

export interface Citation {
  n: number;
  chunk_id: string;
  parent_id: string;
  citation_str: string;
  official_title: string;
  doc_type?: string | null;
  page: number | null;
  lines: [number | null, number | null] | null;
  score: number;
  score_rerank?: number | null;
  skor_bm25?: number | null;
  skor_fusi?: number | null;
  text: string;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
}

export interface AskResponse {
  answer: string;
  citations: Citation[];
  trace: TraceNode[];
  usage: { jumlah_kandidat?: number; detik?: number; context_chars?: number;
           tokens?: TokenUsage };
  mode: string;
  history_id: string;
  reasoning?: string | null;
  reasoning_raw?: string | null;
}

export interface LLMSetting {
  provider: "openai" | "anthropic" | "google" | "zhipu" | "ollama" | "custom";
  base_url: string;
  api_key: string;
  model: string;
  temperature: number;   // 0.0-1.0, null-ish = default provider (0.2)
  maxTokens: number;     // batas jawaban
}

export interface ModelInfo {
  tag: string;
  dim: number | null;
  count: number | null;
  model: string | null;
  aktif: boolean;
  is_gpu: boolean;
}

export interface Settings {
  mode: "fixed" | "agent";
  bahasa: "id" | "en";   // bahasa UI (jawaban LLM tetap ikut prompt)
  paksaCari: boolean;   // agent: wajib ≥1 search_norma (anti jawab dari memori)
  retrieval: "hybrid" | "dense" | "bm25";
  denseMethod: "exact" | "hnsw" | "ivf" | "flat";
  denseEf: number;
  denseNprobe: number;
  gaya: "ringkas" | "standar" | "rinci";
  rerank: boolean;
  topK: number;                    // pool retrieval (1..50)
  injectK: number;                 // dokumen disuntik ke LLM (<= topK)
  docTypes: string[];              // multi-checklist; [] = semua
  year: string;
  embedding: string;               // tag bebas, terdeteksi dari /api/models
  llm: LLMSetting;
  llmEnvKey?: boolean;             // server punya LLM_API_KEY (.env); field key boleh kosong
  dryRun: boolean;
}

export const DOC_TYPES = ["UUD", "TAP MPR", "UU", "PERPPU", "PP", "PERPRES",
  "PERMEN", "PERLEMBAGA", "PERBAWASLU", "KEPPRES"];

export const DEFAULT_SETTINGS: Settings = {
  mode: "agent",
  bahasa: "id",
  paksaCari: true,
  retrieval: "hybrid",
  denseMethod: "exact",              // numpy exact (perilaku lama); Milvus bila perlu
  denseEf: 64,
  denseNprobe: 16,
  gaya: "standar",
  rerank: true,                     // default ON sesuai permintaan
  topK: 8,
  injectK: 8,
  docTypes: [],
  year: "",
  embedding: "qwen3-06b",
  dryRun: false,                    // default OFF = LLM live
  llm: {
    provider: "openai",
    base_url: "",
    api_key: "",
    model: "",
    temperature: 0.2,
    maxTokens: 8192,
  },
};

export interface ChatTurn {
  role: "user" | "bot";
  text: string;
  citations?: Citation[];
  trace?: TraceNode[];
  usage?: AskResponse["usage"];
  mode?: string;
  historyId?: string;
  dryRun?: boolean;
  replayOfNodeId?: string;
  streaming?: boolean;   // true selama SSE berjalan (jawaban mengalir)
  status?: string;       // label tahap terakhir ("Mencari kandidat…")
  reasoning?: string | null;  // penalaran model (bila provider mengirim)
  reasoningRaw?: string | null;  // mentah provider (debug joiner)
}

export interface HistoryMeta {
  id: string; created_at: string; question: string;
  mode: string; retrieval: string;
}
