import { NextResponse } from "next/server";
import { fetchTranscript } from "youtube-transcript-plus";
import { buildChapters, youtubeDescription } from "../../../lib/chapters";

export const maxDuration = 30;

export async function POST(request: Request) {
  try {
    const { url, chapters = 6 } = await request.json();
    if (typeof url !== "string" || url.length > 2048 || !/^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\//i.test(url)) {
      return NextResponse.json({ error: "Вставьте корректную ссылку YouTube" }, { status: 400 });
    }

    const result = await fetchTranscript(url, {
      lang: "ru",
      videoDetails: true,
      retries: 2,
      retryDelay: 500,
      userAgent: "Mozilla/5.0 (compatible; Chapterly/1.0)"
    });
    const transcript = result.segments.map((segment) => segment.text).join(" ").replace(/\s+/g, " ").trim();
    const durationSeconds = Number(result.videoDetails.lengthSeconds) || Math.ceil(
      Math.max(...result.segments.map((segment) => segment.offset + segment.duration), 180)
    );
    const generated = buildChapters(transcript, durationSeconds / 60, Number(chapters));

    return NextResponse.json({
      title: result.videoDetails.title,
      durationMinutes: Math.max(3, Math.round(durationSeconds / 60)),
      transcript,
      chapters: generated,
      description: youtubeDescription(generated),
      engine: "youtube-captions+chapterly-tfidf"
    });
  } catch (error) {
    console.error("YouTube transcript error", error);
    return NextResponse.json(
      { error: "Не удалось получить русские субтитры. Проверьте доступность видео или попробуйте вставить расшифровку." },
      { status: 422 }
    );
  }
}
