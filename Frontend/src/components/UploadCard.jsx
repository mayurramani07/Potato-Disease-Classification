import { motion } from "framer-motion";
import { UploadCloud } from "lucide-react";

export default function UploadCard({ onFileSelect }) {
  return (
    <motion.div
      className="w-full max-w-md p-8 border border-white/10 rounded-2xl bg-white/5 backdrop-blur-xl shadow-lg hover:shadow-green-500/20 transition"
      whileHover={{ scale: 1.05 }}
    >
      <input
        type="file"
        onChange={(e) => onFileSelect(e.target.files[0])}
        className="hidden"
        id="fileUpload"
      />

      <label htmlFor="fileUpload" className="cursor-pointer text-center block">
        <UploadCloud className="mx-auto mb-4 text-green-400" size={40} />
        <p className="text-lg font-semibold">Upload Leaf Image</p>
        <p className="text-sm text-gray-400">Click or drag & drop</p>
      </label>
    </motion.div>
  );
}