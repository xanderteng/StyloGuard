import { View, Text } from "react-native";
import { Link } from "expo-router";
import { SafeAreaView } from "react-native-safe-area-context";

export default function Home() {
  return (
    <SafeAreaView className="flex-1 items-center justify-center bg-gray-50 dark:bg-gray-900">
      <View className="p-6 bg-white dark:bg-gray-800 rounded-2xl shadow-xl shadow-blue-900/20">
        <Text className="text-3xl font-bold text-gray-900 dark:text-white mb-4 text-center">
          🛡️ StyloGuard
        </Text>
        <Text className="text-gray-600 dark:text-gray-300 text-center mb-6">
          Advanced Indonesian Stylometry & Ghostwriter Detection
        </Text>
        
        <Link href="/about" className="bg-blue-600 px-6 py-3 rounded-xl overflow-hidden self-center mt-2">
          <Text className="text-white font-semibold text-center">Get Started</Text>
        </Link>
      </View>
    </SafeAreaView>
  );
}
