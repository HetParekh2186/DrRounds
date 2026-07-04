import { ScrollViewStyleReset } from 'expo-router/html';
import type { PropsWithChildren } from 'react';

/**
 * Root HTML document for the static web export. Only used at build time
 * (see https://docs.expo.dev/router/reference/static-rendering/#root-html) —
 * this is where iOS "Add to Home Screen" needs its meta tags and manifest
 * link, since there's no native Info.plist on web.
 */
export default function Root({ children }: PropsWithChildren) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
        <meta name="theme-color" content="#1f9d7a" />

        <link rel="manifest" href="/manifest.json" />
        <link rel="icon" href="/icon.png" />
        <link rel="apple-touch-icon" href="/icon.png" />

        {/* iOS-specific: makes the home-screen icon launch full-screen, no Safari chrome */}
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="Dr Rounds" />

        <ScrollViewStyleReset />
      </head>
      <body>{children}</body>
    </html>
  );
}
