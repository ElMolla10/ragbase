import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

const LOCALE_KEY = "ragbase.enterprise.locale";

const en = {
  product: "RAGBase Enterprise",
  developmentOnly: "Development only — not production qualification",
  signInTitle: "Sign in with a development identity",
  signInHelp:
    "Paste the bearer token for Alice, Bob, or Viewer from the local development access file. The token stays in this browser tab and is not written to the repository.",
  identityNote:
    "Alice, Bob, and Viewer are synthetic development identities, not company SSO.",
  tokenLabel: "Development token",
  signIn: "Sign in",
  signingIn: "Checking token…",
  signOut: "Sign out",
  signedIn: "Signed in",
  showToken: "Show token",
  hideToken: "Hide token",
  workspaces: "Workspaces",
  documents: "Documents",
  upload: "Upload PDF, TXT, Markdown, or CSV",
  uploadHint: "Up to 5 MB. Native text extraction only. Calculate is a separate action on one ready CSV.",
  uploadBusy: "Uploading…",
  extraction: "Extraction",
  ask: "Ask",
  coverage: "Exhaustive coverage",
  cancel: "Cancel",
  questionLabel: "Question",
  questionPlaceholder: "Ask about the selected versions, or /calculate for one CSV",
  model: "Model",
  provider: "Provider",
  requestedModel: "Requested model",
  notReported: "Not reported",
  contextCoverage: "Context coverage",
  noModelCall: "No model call: no usable extracted context was available.",
  sources: "Sources",
  preview: "Preview source",
  close: "Close",
  gaps: "Gaps",
  covered: "Covered",
  partial: "Partial",
  complete: "Complete",
  selectHint: "Select 1–8 indexed versions.",
  noWorkspaces: "No workspaces for this identity.",
  noDocuments: "No documents yet.",
  externalProcessing:
    "Questions and retrieved text are sent to the configured provider.",
  fixtureNotice: "Development fixture; no live model was called.",
  notCalculation: "This answer is not a spreadsheet calculation.",
  calculate: "Calculate",
  oneCsv: "Choose exactly one ready CSV version.",
  csvOnly: "Calculations run on one ready CSV version.",
  calculationRecord: "Calculation record",
  dataset: "Dataset",
  columnsLabel: "Columns",
  filters: "Filters",
  operation: "Operation",
  units: "Units",
  includedRows: "Included rows",
  excludedRows: "Excluded rows",
  exclusionDetail: "Header {header}, subtotal {subtotal}, filter {filter}",
  noFilters: "None",
  clarificationTitle: "Clarification",
  noNumber: "No number was calculated.",
  downloadCsv: "Download CSV",
  downloadJson: "Download JSON",
  resultTable: "Result table",
  digest: "SHA-256",
  stageCalculating: "Calculating",
  plannerCalled: "The planner ran. The number is the server record, not model text.",
  explicitPlan: "No planner call. The number is the server record.",
  downloadRejected: "That download is not an authenticated calculation export.",
  calculationHint: "Calculate sends one ready CSV question to the server. This page displays the returned record and does not evaluate formulas.",
  retrievalPending: "Indexing is not finished.",
  warnings: "Warnings",
  refresh: "Refresh",
  tooManyVersions: "Choose at most 8 versions.",
  questionRequired: "Enter a question.",
  apiUnavailable: "The enterprise API is unavailable.",
  authFailed: "The development token was not accepted.",
  generation: "Generation",
  sectionProgress: "Covered {covered} of {total} sections",
  selectedCount: "{count} selected",
  passages: "{included} of {total} passages supplied",
  productionUnqualified: "Not production qualified",
  productionQualified: "Payload says production qualified",
  resultTables: "Result tables",
  cancelExtraction: "Cancel extraction",
  loadFailed: "Could not load this workspace.",
  previewTitle: "Source preview",
  nativePreviewNote:
    "Native text only. Layout, tables, images, and OCR are not validated.",
  stageQueued: "Queued",
  stageGenerating: "Generating and validating",
  stageCovering: "Covering sections",
  stageReducing: "Reducing sections",
  statusQueued: "Queued",
  statusExtracting: "Extracting",
  statusExtracted: "Extracted",
  statusNeedsReview: "Needs review",
  statusFailed: "Failed",
  statusCancelled: "Cancelled",
  statusStored: "Stored",
  statusNotQueued: "Not queued",
  emptyChat: "Select indexed versions, then ask, calculate one CSV, or start exhaustive coverage.",
  chatHint:
    "Answers stay bounded unless you start exhaustive coverage. Start a question with /calculate, or use Calculate, for one ready CSV. A complete label is the coverage record, not the model’s wording.",
  partialOverview: "Partial overview; some extracted text was not supplied.",
  scope: "Scope",
  limitation: "Limitation",
  filename: "File",
  version: "Version",
  status: "Status",
  size: "Size",
  location: "Location",
  gapCode: "Gap",
  coverageMap: "Section map",
  page: "Page",
  rowLabel: "Row",
  lines: "Lines",
  ready: "Ready for questions",
  notIndexed: "Not indexed",
  previewUnavailable: "Preview is not available for this source.",
  streamInterrupted: "The stream ended before a complete answer was released.",
  cancelled: "Cancelled.",
  privacyStandard:
    "Standard provider privacy; zero retention is not guaranteed.",
  bytes: "bytes",
  stillProcessing: "Still processing",
  fileRejected: "Choose a PDF, TXT, Markdown, or CSV file of at most 5 MB.",
  calculationNote:
    "The result table shows server text only. Formula-like cells stay literal.",
} as const;

export type MessageKey = keyof typeof en;
export type Locale = "en" | "ar";

const ar: Record<MessageKey, string> = {
  product: "RAGBase للمؤسسات",
  developmentOnly: "للتطوير فقط — ليس تأهيلاً للإنتاج",
  signInTitle: "الدخول بهوية تطوير",
  signInHelp:
    "الصق رمز الدخول الخاص بأليس أو بوب أو العارض من ملف الوصول المحلي. يبقى الرمز في علامة التبويب هذه ولا يُكتب في المستودع.",
  identityNote: "أليس وبوب والعارض هويات تطوير اصطناعية، وليست دخول الشركة الموحّد.",
  tokenLabel: "رمز التطوير",
  signIn: "دخول",
  signingIn: "جارٍ التحقق من الرمز…",
  signOut: "خروج",
  signedIn: "تم الدخول",
  showToken: "إظهار الرمز",
  hideToken: "إخفاء الرمز",
  workspaces: "مساحات العمل",
  documents: "المستندات",
  upload: "ارفع ملف PDF أو TXT أو Markdown أو CSV",
  uploadHint: "الحد 5 ميغابايت. استخراج النص الأصلي فقط. الحساب إجراء منفصل على ملف CSV جاهز واحد.",
  uploadBusy: "جارٍ الرفع…",
  extraction: "الاستخراج",
  ask: "اسأل",
  coverage: "تغطية شاملة",
  cancel: "إلغاء",
  questionLabel: "السؤال",
  questionPlaceholder: "اسأل عن الإصدارات المحددة، أو /calculate لملف CSV واحد",
  model: "النموذج",
  provider: "المزوّد",
  requestedModel: "النموذج المطلوب",
  notReported: "غير مذكور",
  contextCoverage: "تغطية السياق",
  noModelCall: "لم يُستدعَ نموذج: لا يوجد سياق مستخرج صالح.",
  sources: "المصادر",
  preview: "معاينة المصدر",
  close: "إغلاق",
  gaps: "الفجوات",
  covered: "مغطى",
  partial: "جزئي",
  complete: "مكتمل",
  selectHint: "اختر من إصدار واحد إلى ثمانية إصدارات مفهرسة.",
  noWorkspaces: "لا توجد مساحات عمل لهذه الهوية.",
  noDocuments: "لا توجد مستندات بعد.",
  externalProcessing: "تُرسل الأسئلة والنص المسترجع إلى المزوّد المضبوط.",
  fixtureNotice: "مخرجات أداة التطوير؛ لم يُستدعَ نموذج حي.",
  notCalculation: "هذه الإجابة ليست حساب جدول.",
  calculate: "احسب",
  oneCsv: "اختر إصدار CSV جاهزاً واحداً فقط.",
  csvOnly: "يعمل الحساب على إصدار CSV جاهز واحد.",
  calculationRecord: "سجل الحساب",
  dataset: "مجموعة البيانات",
  columnsLabel: "الأعمدة",
  filters: "المرشحات",
  operation: "العملية",
  units: "الوحدات",
  includedRows: "الصفوف المشمولة",
  excludedRows: "الصفوف المستبعدة",
  exclusionDetail: "ترويسة {header}، مجاميع فرعية {subtotal}، مرشح {filter}",
  noFilters: "لا يوجد",
  clarificationTitle: "طلب توضيح",
  noNumber: "لم يُحسب أي رقم.",
  downloadCsv: "تنزيل CSV",
  downloadJson: "تنزيل JSON",
  resultTable: "جدول النتيجة",
  digest: "SHA-256",
  stageCalculating: "جارٍ الحساب",
  plannerCalled: "اشتغل المخطِّط. الرقم من سجل الخادم وليس من نص النموذج.",
  explicitPlan: "لم يُستدعَ المخطِّط. الرقم من سجل الخادم.",
  downloadRejected: "هذا التنزيل ليس تصدير حساب مصادقاً عليه.",
  calculationHint: "يرسل الحساب سؤالاً عن ملف CSV جاهز واحد إلى الخادم. تعرض هذه الصفحة السجل المعاد ولا تقيّم الصيغ.",
  retrievalPending: "الفهرسة لم تنتهِ.",
  warnings: "تحذيرات",
  refresh: "تحديث",
  tooManyVersions: "اختر ثمانية إصدارات كحد أقصى.",
  questionRequired: "اكتب سؤالاً.",
  apiUnavailable: "واجهة المؤسسة غير متاحة.",
  authFailed: "لم يُقبل رمز التطوير.",
  generation: "التوليد",
  sectionProgress: "تمت تغطية {covered} من {total} أقسام",
  selectedCount: "المحدد: {count}",
  passages: "تم توريد {included} من {total} مقطعاً",
  productionUnqualified: "غير مؤهَّل للإنتاج",
  productionQualified: "الحمولة تقول إن الإنتاج مؤهَّل",
  resultTables: "جداول النتائج",
  cancelExtraction: "إلغاء الاستخراج",
  loadFailed: "تعذر تحميل مساحة العمل.",
  previewTitle: "معاينة المصدر",
  nativePreviewNote:
    "نص أصلي فقط. التخطيط والجداول والصور والتعرف الضوئي غير متحقَّق منها.",
  stageQueued: "في الانتظار",
  stageGenerating: "التوليد والتحقق",
  stageCovering: "تغطية الأقسام",
  stageReducing: "تجميع الأقسام",
  statusQueued: "في الانتظار",
  statusExtracting: "جارٍ الاستخراج",
  statusExtracted: "تم الاستخراج",
  statusNeedsReview: "يحتاج مراجعة",
  statusFailed: "فشل",
  statusCancelled: "أُلغي",
  statusStored: "مُخزَّن",
  statusNotQueued: "غير مُدرَج",
  emptyChat: "اختر إصدارات مفهرسة، ثم اسأل أو احسب ملف CSV واحداً أو ابدأ تغطية شاملة.",
  chatHint:
    "تبقى الإجابات محدودة ما لم تبدأ تغطية شاملة. ابدأ السؤال بـ /calculate أو استخدم احسب لملف CSV جاهز واحد. وصف الاكتمال هو سجل التغطية، وليس صياغة النموذج.",
  partialOverview: "نظرة جزئية؛ بعض النص المستخرج لم يُورَّد.",
  scope: "النطاق",
  limitation: "القيد",
  filename: "الملف",
  version: "الإصدار",
  status: "الحالة",
  size: "الحجم",
  location: "الموضع",
  gapCode: "الفجوة",
  coverageMap: "خريطة الأقسام",
  page: "صفحة",
  rowLabel: "صف",
  lines: "أسطر",
  ready: "جاهز للأسئلة",
  notIndexed: "غير مفهرس",
  previewUnavailable: "المعاينة غير متاحة لهذا المصدر.",
  streamInterrupted: "انتهى البث قبل إصدار إجابة مكتملة.",
  cancelled: "أُلغي الطلب.",
  privacyStandard: "خصوصية المزوّد القياسية؛ لا يُضمن عدم الاحتفاظ بالبيانات.",
  bytes: "بايت",
  stillProcessing: "ما زال قيد المعالجة",
  fileRejected: "اختر ملف PDF أو TXT أو Markdown أو CSV لا يتجاوز 5 ميغابايت.",
  calculationNote:
    "جدول النتيجة يعرض نص الخادم فقط. تبقى الخلايا الشبيهة بالصيغ كما هي.",
};

export function formatMessage(
  template: string,
  values: Record<string, string | number>,
): string {
  return template.replace(/\{(\w+)\}/g, (_, key: string) =>
    values[key] === undefined ? "" : String(values[key]),
  );
}

function readLocale(): Locale {
  try {
    return localStorage.getItem(LOCALE_KEY) === "ar" ? "ar" : "en";
  } catch {
    return "en";
  }
}

type LocaleValue = {
  locale: Locale;
  t: (key: MessageKey) => string;
  toggle: () => void;
};

const LocaleContext = createContext<LocaleValue | null>(null);

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState<Locale>(readLocale);
  useEffect(() => {
    document.documentElement.lang = locale === "ar" ? "ar" : "en";
    document.documentElement.dir = locale === "ar" ? "rtl" : "ltr";
    try {
      localStorage.setItem(LOCALE_KEY, locale);
    } catch {
      /* private mode */
    }
  }, [locale]);
  const value = useMemo<LocaleValue>(
    () => ({
      locale,
      t: (key) => (locale === "ar" ? ar[key] : en[key]),
      toggle: () => setLocale((current) => (current === "ar" ? "en" : "ar")),
    }),
    [locale],
  );
  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
}

export function useLocale(): LocaleValue {
  const value = useContext(LocaleContext);
  if (!value) throw new Error("Locale is missing");
  return value;
}

export function statusLabel(status: string, t: (key: MessageKey) => string): string {
  const known: Record<string, MessageKey> = {
    queued: "statusQueued",
    extracting: "statusExtracting",
    extracted: "statusExtracted",
    needs_review: "statusNeedsReview",
    failed: "statusFailed",
    cancelled: "statusCancelled",
    stored: "statusStored",
    not_queued: "statusNotQueued",
  };
  return known[status] ? t(known[status]) : status;
}
