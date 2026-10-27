import type { ElementType, ReactNode } from "react";

export function BidiText({
  text,
  as = "span",
  className = "",
}: {
  text: string;
  as?: ElementType;
  className?: string;
}) {
  if (!text) return null;
  const Tag = as;
  return (
    <Tag className={`bidi ${className}`.trim()} dir="auto" style={{ unicodeBidi: "isolate" }}>
      {text}
    </Tag>
  );
}

export function Isolate({ children }: { children: ReactNode }) {
  return (
    <span className="bidi isolate" dir="ltr" style={{ unicodeBidi: "isolate" }}>
      {children}
    </span>
  );
}
