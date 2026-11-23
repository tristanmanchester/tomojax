"use client";
import { cn } from "cn";
import { buttonVariants } from "fumadocs-ui/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "fumadocs-ui/components/ui/popover";
import { Check, ChevronDown, Copy, ExternalLink, X } from "lucide-react";
import { type ReactNode, useEffect, useRef, useState } from "react";

const markdownCache = new Map<string, string>();

const copyLabels = {
  idle: "Copy as Markdown",
  copying: "Copying…",
  copied: "Copied",
  failed: "Copy failed",
};

/** Copies the page's Markdown, fetched from `markdownPath`, to the clipboard. */
export function CopyMarkdownButton({ markdownPath }: { markdownPath: string }) {
  const [state, setState] = useState<keyof typeof copyLabels>("idle");
  const reset = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(reset.current), []);

  async function copy() {
    window.clearTimeout(reset.current);
    setState("copying");
    try {
      const cached = markdownCache.get(markdownPath);
      // Safari only allows a clipboard write inside the click, so the write
      // starts at once and the fetch goes into the ClipboardItem as a promise.
      await (cached
        ? navigator.clipboard.writeText(cached)
        : navigator.clipboard.write([
            new ClipboardItem({
              "text/plain": fetch(markdownPath).then(async (res) => {
                if (!res.ok) throw new Error(`${markdownPath}: ${res.status}`);
                const text = await res.text();
                markdownCache.set(markdownPath, text);
                // Older Chromium takes a Blob here but not a string.
                return new Blob([text], { type: "text/plain" });
              }),
            }),
          ]));
      setState("copied");
    } catch {
      setState("failed");
    }
    reset.current = window.setTimeout(() => setState("idle"), 2000);
  }

  return (
    <button
      type="button"
      disabled={state === "copying"}
      aria-live="polite"
      onClick={copy}
      className={cn(
        buttonVariants({ variant: "secondary", size: "sm" }),
        "gap-2 whitespace-nowrap [&_svg]:size-3.5 [&_svg]:text-fd-muted-foreground",
      )}
    >
      {state === "copied" ? <Check /> : state === "failed" ? <X /> : <Copy />}
      {copyLabels[state]}
    </button>
  );
}

interface Target {
  name: string;
  href: string;
  icon: ReactNode;
}

function targets(markdownUrl: string, title: string) {
  const page = `${markdownUrl}, the TomoJAX documentation page "${title}"`;
  const question = `Read ${page}. I want to ask questions about it.`;
  const task = `Read ${page}, then help me use TomoJAX as it describes.`;
  // encodeURIComponent writes spaces as %20; some app links read "+" literally.
  const query = (params: Record<string, string>) =>
    Object.entries(params)
      .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
      .join("&");

  const chat: Target[] = [
    {
      name: "Claude",
      href: `https://claude.ai/new?${query({ q: question })}`,
      icon: <ClaudeIcon />,
    },
    {
      name: "ChatGPT",
      href: `https://chatgpt.com/?${query({ prompt: question, hints: "search" })}`,
      icon: <OpenAIIcon />,
    },
  ];
  const agents: Target[] = [
    {
      name: "Claude Code",
      href: `claude-cli://open?${query({ q: task })}`,
      icon: <ClaudeIcon />,
    },
    {
      name: "Codex",
      href: `codex://new?${query({ prompt: task })}`,
      icon: <OpenAIIcon />,
    },
    {
      name: "Cursor",
      href: `https://cursor.com/link/prompt?${query({ text: task })}`,
      icon: <CursorIcon />,
    },
    {
      name: "T3 Code",
      href: `https://app.t3.codes/new?${query({ prompt: task })}`,
      icon: <T3Icon />,
    },
  ];
  return { chat, agents };
}

/**
 * A menu that opens the page in a chat assistant or a coding agent with a
 * prompt that points it at the page's Markdown. Nothing is sent until the
 * reader sends it.
 */
export function AskAIPopover({
  markdownUrl,
  title,
}: {
  markdownUrl: string;
  title: string;
}) {
  const { chat, agents } = targets(markdownUrl, title);

  return (
    <Popover>
      <PopoverTrigger
        className={cn(
          buttonVariants({ variant: "secondary", size: "sm" }),
          "gap-2 whitespace-nowrap data-[popup-open]:bg-fd-accent data-[popup-open]:text-fd-accent-foreground",
        )}
      >
        Ask your AI
        <ChevronDown className="size-3.5 text-fd-muted-foreground" />
      </PopoverTrigger>
      <PopoverContent aria-label="Ask your AI" className="flex flex-col">
        <Group label="Chat" items={chat} />
        <Group label="Coding agent" items={agents} />
      </PopoverContent>
    </Popover>
  );
}

function Group({ label, items }: { label: string; items: Target[] }) {
  return (
    <div className="flex flex-col not-first:mt-2">
      <p className="px-2 pb-1 text-xs font-medium text-fd-muted-foreground">
        {label}
      </p>
      {items.map((item) => {
        // A custom scheme hands off to an app; a new tab would stay blank.
        const web = item.href.startsWith("https:");
        return (
          <a
            key={item.name}
            href={item.href}
            {...(web ? { target: "_blank", rel: "noreferrer noopener" } : {})}
            className="text-sm p-2 rounded-lg inline-flex items-center gap-2 hover:text-fd-accent-foreground hover:bg-fd-accent [&_svg]:size-4"
          >
            {item.icon}
            {item.name}
            {web && (
              <ExternalLink className="text-fd-muted-foreground size-3.5 ms-auto" />
            )}
          </a>
        );
      })}
    </div>
  );
}

/** An icon-only link to the page's source file on GitHub. */
export function GitHubSourceLink({ href }: { href: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      aria-label="View source on GitHub"
      title="View source on GitHub"
      className={cn(
        buttonVariants({ variant: "secondary", size: "sm" }),
        "px-1.5 [&_svg]:size-4",
      )}
    >
      <GitHubIcon />
    </a>
  );
}

function GitHubIcon() {
  return (
    <svg fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
    </svg>
  );
}

function ClaudeIcon() {
  return (
    <svg fill="currentColor" viewBox="0 0 256 257" aria-hidden="true">
      <path d="m50.228 170.321 50.357-28.257.843-2.463-.843-1.361h-2.462l-8.426-.518-28.775-.778-24.952-1.037-24.175-1.296-6.092-1.297L0 125.796l.583-3.759 5.12-3.434 7.324.648 16.202 1.101 24.304 1.685 17.629 1.037 26.118 2.722h4.148l.583-1.685-1.426-1.037-1.101-1.037-25.147-17.045-27.22-18.017-14.258-10.37-7.713-5.25-3.888-4.925-1.685-10.758 7-7.713 9.397.649 2.398.648 9.527 7.323 20.35 15.75L94.817 91.9l3.889 3.24 1.555-1.102.195-.777-1.75-2.917-14.453-26.118-15.425-26.572-6.87-11.018-1.814-6.61c-.648-2.723-1.102-4.991-1.102-7.778l7.972-10.823L71.42 0 82.05 1.426l4.472 3.888 6.61 15.101 10.694 23.786 16.591 32.34 4.861 9.592 2.592 8.879.973 2.722h1.685v-1.556l1.36-18.211 2.528-22.36 2.463-28.776.843-8.1 4.018-9.722 7.971-5.25 6.222 2.981 5.12 7.324-.713 4.73-3.046 19.768-5.962 30.98-3.889 20.739h2.268l2.593-2.593 10.499-13.934 17.628-22.036 7.778-8.749 9.073-9.657 5.833-4.601h11.018l8.1 12.055-3.628 12.443-11.342 14.388-9.398 12.184-13.48 18.147-8.426 14.518.778 1.166 2.01-.194 30.46-6.481 16.462-2.982 19.637-3.37 8.88 4.148.971 4.213-3.5 8.62-20.998 5.184-24.628 4.926-36.682 8.685-.454.324.519.648 16.526 1.555 7.065.389h17.304l32.21 2.398 8.426 5.574 5.055 6.805-.843 5.184-12.962 6.611-17.498-4.148-40.83-9.721-14-3.5h-1.944v1.167l11.666 11.406 21.387 19.314 26.767 24.887 1.36 6.157-3.434 4.86-3.63-.518-23.526-17.693-9.073-7.972-20.545-17.304h-1.36v1.814l4.73 6.935 25.017 37.59 1.296 11.536-1.814 3.76-6.481 2.268-7.13-1.297-14.647-20.544-15.1-23.138-12.185-20.739-1.49.843-7.194 77.448-3.37 3.953-7.778 2.981-6.48-4.925-3.436-7.972 3.435-15.749 4.148-20.544 3.37-16.333 3.046-20.285 1.815-6.74-.13-.454-1.49.194-15.295 20.999-23.267 31.433-18.406 19.702-4.407 1.75-7.648-3.954.713-7.064 4.277-6.286 25.47-32.405 15.36-20.092 9.917-11.6-.065-1.686h-.583L44.07 198.125l-12.055 1.555-5.185-4.86.648-7.972 2.463-2.593 20.35-13.999-.064.065Z" />
    </svg>
  );
}

function OpenAIIcon() {
  return (
    <svg fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z" />
    </svg>
  );
}

function CursorIcon() {
  return (
    <svg fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M11.503.131 1.891 5.678a.84.84 0 0 0-.42.726v11.188c0 .3.162.575.42.724l9.609 5.55a1 1 0 0 0 .998 0l9.61-5.55a.84.84 0 0 0 .42-.724V6.404a.84.84 0 0 0-.42-.726L12.497.131a1.01 1.01 0 0 0-.996 0M2.657 6.338h18.55c.263 0 .43.287.297.515L12.23 22.918c-.062.107-.229.064-.229-.06V12.335a.59.59 0 0 0-.295-.51l-9.11-5.257c-.109-.063-.064-.23.061-.23" />
    </svg>
  );
}

function T3Icon() {
  return (
    <svg fill="currentColor" viewBox="15.5 37 94.4 57" aria-hidden="true">
      <path d="M33.4509 93V47.56H15.5309V37H64.3309V47.56H46.4109V93H33.4509ZM86.7253 93.96C82.832 93.96 78.9653 93.4533 75.1253 92.44C71.2853 91.3733 68.032 89.88 65.3653 87.96L70.4053 78.04C72.5386 79.5867 75.0186 80.8133 77.8453 81.72C80.672 82.6267 83.5253 83.08 86.4053 83.08C89.6586 83.08 92.2186 82.44 94.0853 81.16C95.952 79.88 96.8853 78.12 96.8853 75.88C96.8853 73.7467 96.0586 72.0667 94.4053 70.84C92.752 69.6133 90.0853 69 86.4053 69H80.4853V60.44L96.0853 42.76L97.5253 47.4H68.1653V37H107.365V45.4L91.8453 63.08L85.2853 59.32H89.0453C95.9253 59.32 101.125 60.8667 104.645 63.96C108.165 67.0533 109.925 71.0267 109.925 75.88C109.925 79.0267 109.099 81.9867 107.445 84.76C105.792 87.48 103.259 89.6933 99.8453 91.4C96.432 93.1067 92.0586 93.96 86.7253 93.96Z" />
    </svg>
  );
}
