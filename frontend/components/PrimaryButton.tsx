import { Text, View } from "react-native";

export default function PrimaryButton({ title }: { title: string }) {
  return (
    <View className="bg-blue-600 px-6 py-3 rounded-xl">
      <Text className="text-white font-semibold text-center">{title}</Text>
    </View>
  );
}
