import { describe, expect, it } from "vitest";
import { buildChapters, formatTime, youtubeDescription } from "./chapters";

describe("chapter engine", () => {
  it("formats time like YouTube", () => {
    expect(formatTime(65)).toBe("1:05");
    expect(formatTime(3661)).toBe("1:01:01");
  });
  it("builds ordered chapters and description", () => {
    const text = Array.from({ length: 30 }, (_, i) => i < 10 ? `Маркетинг бренда и аудитория номер ${i}.` : i < 20 ? `Производство подкаста и микрофон номер ${i}.` : `Дистрибуция видео и YouTube номер ${i}.`).join(" ");
    const chapters = buildChapters(text, 60, 4);
    expect(chapters.length).toBeGreaterThanOrEqual(3);
    expect(chapters[0].start).toBe(0);
    expect(youtubeDescription(chapters)).toContain("0:00");
  });
});
