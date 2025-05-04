"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Copy, CheckCircle2 } from "lucide-react";
import { motion } from "framer-motion";
import { ThemeToggle } from "@/components/theme-toggle";

export default function ThaiTextSummarizer() {
  const [inputText, setInputText] = useState("");
  const [summary, setSummary] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleSummarize = async () => {
    if (!inputText.trim()) return;

    setIsLoading(true);
    setError(null);
    setSummary("");

    try {
      // This is where you would make the actual API call to Modal.com
      // For demonstration purposes, I'm simulating an API call with a timeout
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Replace this with your actual API call
      // const response = await fetch('your-modal-api-endpoint', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ text: inputText })
      // })
      //
      // if (!response.ok) throw new Error('Failed to get summary')
      // const data = await response.json()
      // setSummary(data.summary)

      // Simulated response for demonstration
      setSummary(
        `นี่คือตัวอย่างบทสรุปสำหรับข้อความที่คุณป้อน ในการใช้งานจริง บทสรุปจะถูกสร้างโดย API ที่เชื่อมต่อกับ Modal.com`
      );
    } catch (err) {
      setError("เกิดข้อผิดพลาดในการสรุปผล กรุณาลองใหม่");
      console.error("Error fetching summary:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = () => {
    if (!summary) return;
    navigator.clipboard
      .writeText(summary)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch((err) => {
        console.error("Failed to copy:", err);
      });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 dark:from-slate-900 dark:to-blue-950">
      <div className="max-w-5xl mx-auto p-4 md:p-8">
        <motion.header
          className="mb-10 text-center"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="font-outfit text-3xl md:text-4xl font-bold bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent tracking-tight">
            เครื่องมือสรุปข้อความภาษาไทย
          </h1>
          <p className="font-outfit text-slate-600 dark:text-slate-400 mt-2 text-lg font-light tracking-wide">
            Thai Text Summarizer
          </p>
        </motion.header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card className="overflow-hidden border-0 shadow-lg bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm">
              <CardHeader className="bg-gradient-to-r from-blue-500 to-violet-500 text-white">
                <CardTitle className="text-xl font-outfit font-semibold tracking-wide">
                  ป้อนข้อความภาษาไทย
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6">
                <div className="space-y-4">
                  <div className="relative">
                    <textarea
                      className="w-full h-64 p-4 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 shadow-inner focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-y font-noto-thai text-base leading-relaxed"
                      placeholder="วางข้อความภาษาไทยที่นี่..."
                      value={inputText}
                      onChange={(e) => setInputText(e.target.value)}
                    />
                    <div className="absolute bottom-3 right-3 text-xs text-slate-400 font-outfit">
                      {inputText.length} ตัวอักษร
                    </div>
                  </div>
                  <Button
                    onClick={handleSummarize}
                    disabled={isLoading || !inputText.trim()}
                    className="w-full py-6 bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-700 hover:to-violet-700 text-white font-outfit font-semibold rounded-xl shadow-md transition-all duration-200 hover:shadow-lg"
                  >
                    {isLoading ? (
                      <div className="flex items-center justify-center">
                        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                        <span className="font-noto-thai">กำลังสรุปผล...</span>
                      </div>
                    ) : (
                      <span className="font-noto-thai text-lg tracking-wide">
                        สรุปข้อความ
                      </span>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Output Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="overflow-hidden border-0 shadow-lg bg-white/80 dark:bg-slate-800/50 backdrop-blur-sm h-full">
              <CardHeader className="bg-gradient-to-r from-violet-500 to-purple-500 text-white flex flex-row justify-between items-center">
                <CardTitle className="text-xl font-outfit font-semibold tracking-wide">
                  บทสรุป
                </CardTitle>
                {summary && (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={copyToClipboard}
                    className="text-white hover:text-white/80 hover:bg-white/10 font-outfit"
                  >
                    {copied ? (
                      <CheckCircle2 className="h-5 w-5" />
                    ) : (
                      <Copy className="h-5 w-5" />
                    )}
                  </Button>
                )}
              </CardHeader>
              <CardContent className="p-6">
                <div className="w-full min-h-[16rem] rounded-xl bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm">
                  {isLoading ? (
                    <div className="flex flex-col items-center justify-center h-64 text-slate-500">
                      <div className="relative w-16 h-16">
                        <div className="absolute inset-0 flex items-center justify-center">
                          <Loader2 className="h-10 w-10 animate-spin text-blue-500" />
                        </div>
                        <div className="absolute inset-0 rounded-full border-t-2 border-blue-500 animate-ping opacity-20"></div>
                      </div>
                      <p className="mt-4 text-slate-500 dark:text-slate-400 font-noto-thai">
                        กำลังสรุปผล โปรดรอสักครู่...
                      </p>
                    </div>
                  ) : error ? (
                    <motion.div
                      className="flex items-center justify-center h-64 text-red-500 p-4 text-center"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.3 }}
                    >
                      <p className="font-noto-thai">{error}</p>
                    </motion.div>
                  ) : summary ? (
                    <motion.div
                      className="p-6 h-64 overflow-auto"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.3 }}
                    >
                      <p className="whitespace-pre-line text-slate-700 dark:text-slate-300 leading-relaxed font-noto-thai text-base">
                        {summary}
                      </p>
                    </motion.div>
                  ) : (
                    <div className="flex items-center justify-center h-64 text-slate-400 dark:text-slate-500">
                      <p className="font-noto-thai">บทสรุปจะแสดงที่นี่...</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
      <ThemeToggle />
    </div>
  );
}
