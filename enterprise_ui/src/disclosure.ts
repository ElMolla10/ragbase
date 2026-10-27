import type { AnswerPayload, SectionRow } from "./types";

export type Disclosure = {
  model: string | null;
  provider: string | null;
  requestedModel: string | null;
  fixture: boolean;
  externalProcessing: boolean;
  productionQualified: boolean | null;
  scopeNotice: string | null;
  included: number | null;
  total: number | null;
  allIncluded: boolean | null;
  modelCalled: boolean | null;
  privacyPolicy: string | null;
};

function textOrNull(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function readDisclosure(answer: AnswerPayload): Disclosure {
  const metadata = answer.metadata ?? {};
  const model =
    metadata.model && typeof metadata.model === "object"
      ? (metadata.model as Record<string, unknown>)
      : {};
  const route =
    model.resolved_route && typeof model.resolved_route === "object"
      ? (model.resolved_route as Record<string, unknown>)
      : {};
  const coverage =
    metadata.coverage && typeof metadata.coverage === "object"
      ? (metadata.coverage as Record<string, unknown>)
      : {};
  const qualified =
    typeof model.production_qualified === "boolean"
      ? model.production_qualified
      : metadata.production_qualified;
  return {
    model: textOrNull(route.model),
    provider: textOrNull(route.provider),
    requestedModel: textOrNull(model.requested_model),
    fixture: metadata.fixture === true,
    externalProcessing: metadata.external_processing === true,
    productionQualified: typeof qualified === "boolean" ? qualified : null,
    scopeNotice: textOrNull(answer.scope_notice),
    included: numberOrNull(coverage.included_chunks),
    total: numberOrNull(coverage.total_document_chunks),
    allIncluded:
      typeof coverage.all_extracted_chunks_included === "boolean"
        ? coverage.all_extracted_chunks_included
        : null,
    modelCalled: typeof metadata.model_called === "boolean" ? metadata.model_called : null,
    privacyPolicy: textOrNull(metadata.privacy_policy),
  };
}

export function coverageIsComplete(payload: {
  label?: string;
  coverage_complete?: boolean;
}): boolean {
  return payload.label === "complete" && payload.coverage_complete === true;
}

export function locationParts(location: unknown): {
  kind: "page" | "row" | "lines" | "unknown";
  page?: string;
  row?: string;
  start?: string;
  end?: string;
} {
  if (!location || typeof location !== "object") return { kind: "unknown" };
  const row = location as Record<string, unknown>;
  if (row.page !== undefined && row.page !== null) {
    return { kind: "page", page: String(row.page) };
  }
  if (row.row !== undefined && row.row !== null) {
    return { kind: "row", row: String(row.row) };
  }
  if (row.line_start !== undefined || row.line_end !== undefined) {
    return {
      kind: "lines",
      start: row.line_start === undefined ? "?" : String(row.line_start),
      end: row.line_end === undefined ? "?" : String(row.line_end),
    };
  }
  return { kind: "unknown" };
}

export function sectionKey(section: SectionRow, index: number): string {
  return [
    section.version_id ?? "version",
    section.ordinal ?? "ordinal",
    section.part ?? "part",
    section.gap_code ?? "",
    index,
  ].join(":");
}

export function previewPath(value: unknown): string | null {
  if (typeof value !== "string" || !value.startsWith("/api/") || value.includes("..")) {
    return null;
  }
  if (value.includes("\\") || value.includes(" ")) return null;
  try {
    const url = new URL(value, "http://enterprise.local");
    if (url.origin !== "http://enterprise.local" || !url.pathname.startsWith("/api/")) {
      return null;
    }
    return `${url.pathname}${url.search}`;
  } catch {
    return null;
  }
}
