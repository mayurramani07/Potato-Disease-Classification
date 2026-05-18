import { motion } from "framer-motion";

export default function Loader() {
  return (
    <motion.div
      className="mt-6 text-center"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
    >
      <div className="w-10 h-10 border-4 border-green-400 border-t-transparent rounded-full animate-spin mx-auto"></div>
      <p className="mt-2 text-gray-400">Analyzing Leaf...</p>
    </motion.div>
  );
}