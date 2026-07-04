// eslint-disable-next-line @typescript-eslint/no-var-requires
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// expo-sqlite's web backend loads a wa-sqlite .wasm binary; Metro needs to
// treat it as an asset (not source) for the web export to resolve it.
config.resolver.assetExts.push('wasm');

module.exports = config;
