import { buildRootLlmsTxt } from "@/lib/llms-index";

export const revalidate = false;

export async function GET() {
  return new Response(buildRootLlmsTxt(), {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
}
