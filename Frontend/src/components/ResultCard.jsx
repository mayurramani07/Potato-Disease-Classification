import { motion } from "framer-motion";

export default function ResultCard({ result }) {
  const confidence = (result.confidence * 100).toFixed(2);

  const getColor = () => {
    if (confidence > 80) return "bg-green-400";
    if (confidence > 50) return "bg-yellow-400";
    return "bg-red-400";
  };

  const getGlow = () => {
    if (confidence > 80) return "shadow-green-500/30";
    if (confidence > 50) return "shadow-yellow-500/30";
    return "shadow-red-500/30";
  };

  return (
    <motion.div
      className={`mt-8 p-6 rounded-2xl bg-white/5 backdrop-blur-xl border border-white/10 shadow-2xl w-full max-w-md text-center ${getGlow()}`}
      initial={{ y: 50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
    >
      <h2 className="text-xl font-semibold mb-4 tracking-wide text-gray-600">
        🌿 Prediction Result
      </h2>


      <p className="text-3xl font-bold text-green-400 mb-2">
        {result.class}
      </p>


      <div className="inline-block px-4 py-1 rounded-full bg-white/10 text-sm text-gray-600 mb-4">
        Confidence: {confidence}%
      </div>

      <div className="w-full bg-gray-700/50 rounded-full h-3 overflow-hidden">
        <motion.div
          className={`h-3 rounded-full ${getColor()}`}
          initial={{ width: 0 }}
          animate={{ width: `${confidence}%` }}
          transition={{ duration: 0.8 }}
        />
      </div>

      <p className="mt-3 text-sm text-gray-400">
        Model confidence level
      </p>
    </motion.div>
  );
}