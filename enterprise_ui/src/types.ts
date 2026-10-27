export type Workspace = { id: string; name: string };

export type DocumentRecord = {
  id: string;
  filename: string;
  version_id: string;
  version_number: number;
  byte_size: number;
  sha256: string;
  status: string;
  ingestion_status: string;
};

export type ExtractionRecord = {
  status: string;
  error_code?: string | null;
  error_detail?: string | null;
  warnings?: string[];
  extractor?: string | null;
  retrieval_ready?: boolean;
};

export type Claim = {
  source_id?: string;
  quote?: string;
  filename?: string;
  location?: Record<string, unknown> | null;
  preview_url?: string | null;
  warnings?: string[];
  sha256?: string;
};

export type SectionRow = {
  version_id?: string;
  ordinal?: number | null;
  part?: number | null;
  status?: string;
  gap_code?: string | null;
  location?: Record<string, unknown> | null;
  filename?: string;
  sha256?: string;
};

export type AnswerPayload = {
  event: "answer";
  mode?: string;
  insufficient?: boolean;
  answer?: string;
  claims?: Claim[];
  scope_notice?: string;
  limitation?: string;
  downloads?: { table?: string; record?: string };
  coverage_complete?: boolean;
  label?: string;
  sections?: SectionRow[];
  gaps?: SectionRow[];
  metadata?: Record<string, unknown>;
  calculations?: unknown;
};

export type CoveragePayload = {
  event: "coverage";
  run_id?: string;
  coverage_complete?: boolean;
  label?: string;
  scope_notice?: string;
  sections?: SectionRow[];
  gaps?: SectionRow[];
};

export type StreamEvent =
  | {
      event: "queued";
      run_id?: string;
      external_processing?: boolean;
      privacy_policy?: string;
    }
  | { event: "stage"; stage?: string }
  | { event: "section"; covered?: number; total?: number }
  | AnswerPayload
  | CoveragePayload
  | { event: "done"; run_id?: string }
  | { event: "error"; code?: string; detail?: string };

export type HealthReady = {
  status?: string;
  generation?: string;
  documents?: string;
};

export type CalculationResponse = {
  calculation_id?: string;
  status?: string;
  answer?: string;
  record?: unknown;
  citations?: unknown[];
  downloads?: { table?: string; record?: string };
  model_called?: boolean;
  production_qualified?: boolean;
};

export type HealthGeneration = {
  mode?: string;
  production_qualified?: boolean;
  credential_configured?: boolean;
  external_processing?: boolean;
};
