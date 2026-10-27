/**
 * Read a calculation record the server already produced.
 * These helpers display text. They do not evaluate formulas or aggregates.
 */

const UUID =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const DOWNLOAD =
  /^\/api\/workspaces\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\/calculations\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\/(?:table\.csv|record\.json)$/i;

export type CalculationFilter = {
  column: string;
  operator: string;
  value: string;
};

export type CalculationRecordView = {
  status: string;
  code: string | null;
  answer: string;
  filename: string;
  versionId: string;
  sha256: string;
  columns: { name: string; type: string }[];
  filters: CalculationFilter[];
  operation: string;
  units: string;
  included: number | null;
  excluded: number | null;
  headerExcluded: number | null;
  subtotalExcluded: number | null;
  filterExcluded: number | null;
  tableColumns: string[];
  tableRows: string[][];
  limitation: string | null;
  formulaNote: string | null;
  clarification: boolean;
};

export type CalculationDownloads = {
  table: string | null;
  record: string | null;
};

function text(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function count(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function cellText(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

export function isCsvFilename(name: string): boolean {
  const base = name.split(/[/\\]/).pop() ?? name;
  return base.length > 4 && base.toLowerCase().endsWith(".csv");
}

export function stripCalculate(question: string): string {
  const trimmed = question.trim();
  if (trimmed === "/calculate" || trimmed.startsWith("/calculate ")) {
    return trimmed.slice("/calculate".length).trim();
  }
  return trimmed;
}

export function calculationId(metadata: unknown): string | null {
  if (!metadata || typeof metadata !== "object") return null;
  const id = (metadata as { calculation_id?: unknown }).calculation_id;
  return typeof id === "string" && UUID.test(id) ? id : null;
}

export function calculationDownloadPath(value: unknown): string | null {
  if (typeof value !== "string") return null;
  if (value.includes("..") || value.includes("\\") || value.includes("?") || value.includes("#")) {
    return null;
  }
  return DOWNLOAD.test(value) ? value : null;
}

export function readDownloads(value: unknown): CalculationDownloads {
  if (!value || typeof value !== "object") return { table: null, record: null };
  const row = value as Record<string, unknown>;
  return {
    table: calculationDownloadPath(row.table),
    record: calculationDownloadPath(row.record),
  };
}

export function readCalculationRecord(
  value: unknown,
  status?: string,
): CalculationRecordView | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const dataset = record.dataset;
  if (!dataset || typeof dataset !== "object") return null;
  const data = dataset as Record<string, unknown>;
  const filename = text(data.filename);
  const versionId = text(data.version_id);
  if (!filename || !versionId) return null;
  if (!Array.isArray(data.columns)) return null;
  const columns: { name: string; type: string }[] = [];
  for (const column of data.columns) {
    if (!column || typeof column !== "object") return null;
    const row = column as Record<string, unknown>;
    if (typeof row.name !== "string") return null;
    columns.push({ name: row.name, type: typeof row.type === "string" ? row.type : "" });
  }
  const filters: CalculationFilter[] = [];
  if (record.filters !== undefined && record.filters !== null) {
    if (!Array.isArray(record.filters)) return null;
    for (const filter of record.filters) {
      if (!filter || typeof filter !== "object") return null;
      const row = filter as Record<string, unknown>;
      if (typeof row.column !== "string" || typeof row.operator !== "string") return null;
      const listed = Array.isArray(row.values) ? row.values.map(cellText).join(", ") : "";
      filters.push({
        column: row.column,
        operator: row.operator,
        value: row.value === null || row.value === undefined ? listed : cellText(row.value),
      });
    }
  }
  const table = readTable(record.result);
  if (table === null) return null;
  const exclusions =
    record.exclusions && typeof record.exclusions === "object"
      ? (record.exclusions as Record<string, unknown>)
      : {};
  const resolved = status ?? text(record.status) ?? "";
  return {
    status: resolved,
    code: text(record.code),
    answer: text(record.answer) ?? "",
    filename,
    versionId,
    sha256: text(data.sha256) ?? "",
    columns,
    filters,
    operation: formatOperation(record.operation),
    units: text(record.units) ?? "",
    included: count(record.included_row_count),
    excluded: count(record.excluded_row_count),
    headerExcluded: count(exclusions.header),
    subtotalExcluded: count(exclusions.subtotal),
    filterExcluded: count(exclusions.filter),
    tableColumns: table.columns,
    tableRows: table.rows,
    limitation: text(record.limitation),
    formulaNote: text(record.formula_note),
    clarification: resolved === "clarification",
  };
}

function readTable(value: unknown): { columns: string[]; rows: string[][] } | null {
  if (value === null || value === undefined) return { columns: [], rows: [] };
  if (!value || typeof value !== "object") return null;
  const table = value as Record<string, unknown>;
  if (!Array.isArray(table.columns) || !table.columns.every((item) => typeof item === "string")) {
    return null;
  }
  if (!Array.isArray(table.rows)) return null;
  const columns = table.columns;
  const rows: string[][] = [];
  for (const row of table.rows) {
    if (!row || typeof row !== "object" || Array.isArray(row)) return null;
    const cells = row as Record<string, unknown>;
    rows.push(columns.map((column) => cellText(cells[column])));
  }
  return { columns, rows };
}

function formatOperation(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (!value || typeof value !== "object") return "";
  const operation = value as Record<string, unknown>;
  const parts: string[] = [];
  const aggregates = Array.isArray(operation.aggregates) ? operation.aggregates : [];
  const rendered = aggregates
    .map((item) => {
      if (!item || typeof item !== "object") return "";
      const row = item as Record<string, unknown>;
      if (typeof row.function !== "string" || !row.function) return "";
      const column = typeof row.column === "string" && row.column ? row.column : "*";
      return `${row.function}(${column})`;
    })
    .filter(Boolean);
  if (rendered.length > 0) parts.push(rendered.join(", "));
  if (
    Array.isArray(operation.group_by) &&
    operation.group_by.every((item) => typeof item === "string") &&
    operation.group_by.length > 0
  ) {
    parts.push(`group by ${operation.group_by.join(", ")}`);
  }
  if (
    rendered.length === 0 &&
    Array.isArray(operation.project) &&
    operation.project.every((item) => typeof item === "string") &&
    operation.project.length > 0
  ) {
    parts.push(`project ${operation.project.join(", ")}`);
  }
  if (typeof operation.limit === "number") parts.push(`limit ${operation.limit}`);
  return parts.join("; ");
}
