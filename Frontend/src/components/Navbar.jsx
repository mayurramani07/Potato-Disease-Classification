import { Leaf } from "lucide-react";

export default function Navbar() {
  return (
    <div className="w-full p-4 flex justify-between items-center bg-white/5 backdrop-blur-xl border-b border-white/10 shadow-lg">
      <h1 className="text-xl font-bold flex items-center gap-2 text-green-400">
        <Leaf />
        AgroVision
      </h1>
    </div>
  );
}