import { useColorScheme as useNativewindColorScheme } from "nativewind";

/**
 * A custom hook to interact with NativeWind's color scheme.
 * It provides the current color scheme and a function to toggle it.
 */
export function useColorScheme() {
  const { colorScheme, setColorScheme, toggleColorScheme } = useNativewindColorScheme();
  
  return {
    colorScheme: colorScheme ?? 'light',
    isDarkColorScheme: colorScheme === 'dark',
    setColorScheme,
    toggleColorScheme,
  };
}
