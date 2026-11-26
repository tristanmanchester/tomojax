import { buildGuidesLlmsTxt } from "@/lib/llms-index";

export const revalidate = false;

export async function GET() {
  return new Response(buildGuidesLlmsTxt(), {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
}
