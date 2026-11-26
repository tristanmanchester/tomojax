import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";
import Image from "next/image";
import { appName, gitConfig } from "./shared";

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <span className="inline-flex items-center">
          <Image
            src="/brand/tomojax-logo.svg"
            alt={appName}
            width={101}
            height={28}
            priority
            className="h-7 w-auto dark:hidden"
          />
          <Image
            src="/brand/tomojax-logo-dark.svg"
            alt={appName}
            width={101}
            height={28}
            priority
            className="hidden h-7 w-auto dark:block"
          />
        </span>
      ),
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
