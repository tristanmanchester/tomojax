import { buildApiLlmsTxt } from "@/lib/llms-index";

export const revalidate = false;

export async function GET() {
  return new Response(buildApiLlmsTxt(), {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
}
