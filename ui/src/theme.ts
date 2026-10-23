import { createDarkTheme, createLightTheme } from '@fluentui/react-components';
import type { BrandVariants, Theme } from '@fluentui/react-components';

/**
 * Windows 11 brand ramp around #0F6CBD. Fluent derives all component states
 * (hover, pressed, selected, focus) from these stops.
 */
const brandRamp: BrandVariants = {
  10: '#061724',
  20: '#082338',
  30: '#0a2e4a',
  40: '#0c3b5e',
  50: '#0d4a73',
  60: '#0e5389',
  70: '#0f5ca3',
  80: '#0f6cbd',
  90: '#2886de',
  100: '#479ef5',
  110: '#62abf5',
  120: '#77b7f7',
  130: '#96c6fa',
  140: '#b4d6fa',
  150: '#cfe4fc',
  160: '#e8f2fd',
};

const fontFamilyBase =
  '"Segoe UI Variable Text", "Segoe UI Variable", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif';

const fontFamilyMonospace =
  '"Cascadia Mono", "Cascadia Code", Consolas, "Segoe UI Mono", ui-monospace, monospace';

function withTypography(theme: Theme): Theme {
  return {
    ...theme,
    fontFamilyBase,
    fontFamilyMonospace,
    fontFamilyNumeric: fontFamilyBase,
  };
}

export const lightTheme = withTypography(createLightTheme(brandRamp));
export const darkTheme = withTypography(createDarkTheme(brandRamp));
