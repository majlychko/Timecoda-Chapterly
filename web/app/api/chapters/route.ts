import { NextResponse } from "next/server";
import { buildChapters, youtubeDescription } from "../../../lib/chapters";

export async function POST(request: Request) {
  try {
    const { transcript, durationMinutes = 60, chapters = 6 } = await request.json();
    if (typeof transcript !== "string" || transcript.length > 150000) {
      return NextResponse.json({ error: "Некорректная или слишком длинная расшифровка" }, { status: 400 });
    }
    const result = buildChapters(transcript, Number(durationMinutes), Number(chapters));
    return NextResponse.json({ chapters: result, description: youtubeDescription(result), engine: "chapterly-tfidf-mvp" });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Не удалось обработать текст" }, { status: 400 });
  }
}
