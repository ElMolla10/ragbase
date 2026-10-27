/**
 * Renders string cells supplied by the server.
 * This component does not evaluate formulas or aggregates.
 */
export type CalculationTable = {
  id: string;
  title: string;
  columns: string[];
  rows: string[][];
  note?: string;
};

export function readCalculationTables(value: unknown): CalculationTable[] | null {
  if (!value || typeof value !== "object") return null;
  const tables = (value as { tables?: unknown }).tables;
  if (!Array.isArray(tables) || tables.length === 0) return null;
  const parsed: CalculationTable[] = [];
  for (const table of tables) {
    if (!table || typeof table !== "object") return null;
    const row = table as Record<string, unknown>;
    if (typeof row.id !== "string" || typeof row.title !== "string") return null;
    if (!Array.isArray(row.columns) || !row.columns.every((cell) => typeof cell === "string")) {
      return null;
    }
    if (
      !Array.isArray(row.rows) ||
      !row.rows.every(
        (line) => Array.isArray(line) && line.every((cell) => typeof cell === "string"),
      )
    ) {
      return null;
    }
    parsed.push({
      id: row.id,
      title: row.title,
      columns: row.columns,
      rows: row.rows as string[][],
      note: typeof row.note === "string" ? row.note : undefined,
    });
  }
  return parsed;
}

export function ResultTables({ tables }: { tables: CalculationTable[] | null }) {
  if (!tables?.length) return null;
  return (
    <div className="result-tables">
      {tables.map((table) => (
        <figure key={table.id}>
          <figcaption>
            <span className="bidi" dir="auto" style={{ unicodeBidi: "isolate" }}>
              {table.title}
            </span>
          </figcaption>
          <table>
            <thead>
              <tr>
                {table.columns.map((column) => (
                  <th key={column} scope="col">
                    <span className="bidi" dir="auto" style={{ unicodeBidi: "isolate" }}>
                      {column}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((line, index) => (
                <tr key={`${table.id}-${index}`}>
                  {line.map((cell, cellIndex) => (
                    <td key={`${table.id}-${index}-${cellIndex}`}>
                      <span className="bidi" dir="auto" style={{ unicodeBidi: "isolate" }}>
                        {cell}
                      </span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {table.note ? (
            <p className="bidi" dir="auto" style={{ unicodeBidi: "isolate" }}>
              {table.note}
            </p>
          ) : null}
        </figure>
      ))}
    </div>
  );
}
