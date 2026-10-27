import { ResultTables } from "./ResultTables";
import type { CalculationDownloads, CalculationRecordView } from "./record";
import { BidiText, Isolate } from "../bidi";
import { formatMessage, useLocale } from "../i18n";

export function CalculationView({
  record,
  downloads,
  onDownload,
}: {
  record: CalculationRecordView;
  downloads?: CalculationDownloads | null;
  onDownload: (path: string) => void;
}) {
  const { t } = useLocale();
  const table =
    record.clarification || record.tableColumns.length === 0
      ? null
      : [
          {
            id: record.versionId,
            title: t("resultTable"),
            columns: record.tableColumns,
            rows: record.tableRows,
            note: record.formulaNote ?? undefined,
          },
        ];
  const exclusion =
    record.headerExcluded === null &&
    record.subtotalExcluded === null &&
    record.filterExcluded === null
      ? ""
      : formatMessage(t("exclusionDetail"), {
          header: record.headerExcluded ?? 0,
          subtotal: record.subtotalExcluded ?? 0,
          filter: record.filterExcluded ?? 0,
        });
  return (
    <section className="calculation" aria-label={t("calculationRecord")}>
      <p className={record.clarification ? "badge warn" : "badge ok"}>
        {record.clarification ? t("clarificationTitle") : t("calculationRecord")}
      </p>
      {record.clarification ? <p>{t("noNumber")}</p> : null}
      {record.code ? (
        <p>
          <Isolate>{record.code}</Isolate>
        </p>
      ) : null}
      {record.answer ? <BidiText as="div" className="prose" text={record.answer} /> : null}
      <dl className="disclosure">
        <div>
          <dt>{t("dataset")}</dt>
          <dd>
            <BidiText text={record.filename} />
          </dd>
        </div>
        <div>
          <dt>{t("version")}</dt>
          <dd>
            <Isolate>{record.versionId}</Isolate>
          </dd>
        </div>
        {record.sha256 ? (
          <div>
            <dt>{t("digest")}</dt>
            <dd>
              <Isolate>{record.sha256}</Isolate>
            </dd>
          </div>
        ) : null}
        <div>
          <dt>{t("columnsLabel")}</dt>
          <dd>
            {record.columns.length === 0
              ? t("notReported")
              : record.columns.map((column) => (
                  <span key={`${column.name}-${column.type}`} className="column-chip">
                    <BidiText text={column.name} />
                    {column.type ? (
                      <>
                        {" "}
                        <Isolate>({column.type})</Isolate>
                      </>
                    ) : null}
                  </span>
                ))}
          </dd>
        </div>
        <div>
          <dt>{t("filters")}</dt>
          <dd>
            {record.filters.length === 0 ? (
              t("noFilters")
            ) : (
              record.filters.map((filter) => (
                <span key={`${filter.column}-${filter.operator}-${filter.value}`} className="column-chip">
                  <BidiText text={`${filter.column} ${filter.operator} ${filter.value}`} />
                </span>
              ))
            )}
          </dd>
        </div>
        <div>
          <dt>{t("operation")}</dt>
          <dd>{record.operation ? <BidiText text={record.operation} /> : t("notReported")}</dd>
        </div>
        <div>
          <dt>{t("units")}</dt>
          <dd>{record.units ? <BidiText text={record.units} /> : t("notReported")}</dd>
        </div>
        <div>
          <dt>{t("includedRows")}</dt>
          <dd>
            <Isolate>{record.included === null ? t("notReported") : String(record.included)}</Isolate>
          </dd>
        </div>
        <div>
          <dt>{t("excludedRows")}</dt>
          <dd>
            <Isolate>{record.excluded === null ? t("notReported") : String(record.excluded)}</Isolate>
            {exclusion ? <span> · {exclusion}</span> : null}
          </dd>
        </div>
      </dl>
      {record.limitation ? (
        <p className="note">
          <BidiText text={record.limitation} />
        </p>
      ) : null}
      <ResultTables tables={table} />
      <div className="actions">
        {downloads?.table ? (
          <button type="button" className="secondary" onClick={() => onDownload(downloads.table!)}>
            {t("downloadCsv")}
          </button>
        ) : null}
        {downloads?.record ? (
          <button type="button" className="secondary" onClick={() => onDownload(downloads.record!)}>
            {t("downloadJson")}
          </button>
        ) : null}
      </div>
    </section>
  );
}
