import { AppShell } from "@/components/AppShell";
import { Button } from "@/components/ui/primitives";
import Link from "next/link";

export function Placeholder({ title }: { title: string }) {
  return (
    <AppShell title={title}>
      <div className="max-w-[520px] mx-auto my-16 text-center">
        <div className="text-[40px] mb-3 opacity-60">◷</div>
        <h1 className="text-[1.5rem] font-semibold">{title}</h1>
        <p className="text-ink-2 mt-2">
          This module follows the same design system and connects to the same backend. The five
          priority screens — Dashboard, Inventory, Logistics, Knowledge, and Reports — are built out.
        </p>
        <Link href="/" className="inline-block mt-4"><Button variant="secondary">← Back to Control Tower</Button></Link>
      </div>
    </AppShell>
  );
}
