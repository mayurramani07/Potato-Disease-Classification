import { useState, useEffect } from "react";
import UploadCard from "../components/UploadCard";
import ResultCard from "../components/ResultCard";
import Loader from "../components/Loader";
import { predictImage } from "../services/api";
import { motion } from "framer-motion";

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const savedImage = localStorage.getItem("image");
    const savedResult = localStorage.getItem("result");

    if (savedImage) setPreview(savedImage);
    if (savedResult) setResult(JSON.parse(savedResult));
  }, []);

  const handleUpload = async (file) => {
    setFile(file);

    const reader = new FileReader();
    reader.readAsDataURL(file);

    reader.onloadend = async () => {
      const base64 = reader.result;

      // SAVE IMAGE
      localStorage.setItem("image", base64);

      setPreview(base64);
      setLoading(true);
      setResult(null);

      try {
        const res = await predictImage(file);
        setResult(res);


        localStorage.setItem("result", JSON.stringify(res));
      } catch (err) {
        alert("Backend error");
      }

      setLoading(false);
    };
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);

    localStorage.removeItem("image");
    localStorage.removeItem("result");
  };

  return (
    <div className="flex flex-col items-center mt-20 px-4">

      {!preview && <UploadCard onFileSelect={handleUpload} />}

      {preview && (
        <div className="flex flex-col items-center mt-6">
          <motion.img
            src={preview}
            alt="preview"
            className="w-72 h-72 object-cover rounded-xl shadow-lg"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          />

          <button
            onClick={handleClear}
            className="mt-4 px-5 py-2 bg-red-500/80 hover:bg-red-600 text-white rounded-lg transition"
          >
            Remove Image
          </button>
        </div>
      )}

      {loading && <Loader />}
      {result && <ResultCard result={result} />}
    </div>
  );
}