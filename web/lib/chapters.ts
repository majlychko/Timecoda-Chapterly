export type Chapter = { start: number; end: number; title: string; summary: string };

const STOP = new Set("и в во не что он на я с со как а то все она так его но да ты к у уже вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну ли если или быть был до вас там потом для мы их чем была сам без будет кто этот того потому этого какой здесь один сейчас были зачем можно при больше через про всего них какая много хорошо свою перед иногда лучше такой им более всегда между это вообще просто очень типа короче значит знаете сказать который которая которые давайте сегодня будем говорить поэтому собственно например самом деле понятно слушайте спасибо привет всем добрый время суток".split(" "));

function tokens(text: string) {
  return (text.toLowerCase().replace(/ё/g, "е").match(/[a-zа-я0-9]{3,}/gi) || []).filter((w) => !STOP.has(w));
}

function vector(text: string) {
  const map = new Map<string, number>();
  for (const word of tokens(text)) map.set(word, (map.get(word) || 0) + 1);
  return map;
}

function cosine(a: Map<string, number>, b: Map<string, number>) {
  let dot = 0, aa = 0, bb = 0;
  for (const value of a.values()) aa += value * value;
  for (const value of b.values()) bb += value * value;
  for (const [key, value] of a) dot += value * (b.get(key) || 0);
  return dot / Math.max(Math.sqrt(aa * bb), 1e-9);
}

function titleFor(text: string) {
  const words = tokens(text);
  const counts = new Map<string, number>();
  for (const word of words) counts.set(word, (counts.get(word) || 0) + 1);
  const top = [...counts].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([word]) => word);
  if (!top.length) return "Новая тема";
  const title = top.join(" · ");
  return title.charAt(0).toUpperCase() + title.slice(1);
}

function splitSentences(text: string) {
  return text.replace(/\s+/g, " ").trim().split(/(?<=[.!?])\s+/).filter(Boolean);
}

export function formatTime(seconds: number) {
  const n = Math.max(0, Math.floor(seconds));
  const h = Math.floor(n / 3600), m = Math.floor((n % 3600) / 60), s = n % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
}

export function buildChapters(text: string, durationMinutes: number, desired = 6): Chapter[] {
  const sentences = splitSentences(text);
  if (sentences.length < 3) throw new Error("Добавьте расшифровку хотя бы из трёх предложений.");
  const duration = Math.max(180, durationMinutes * 60);
  const blockCount = Math.min(Math.max(desired * 3, 12), sentences.length);
  const blocks = Array.from({ length: blockCount }, (_, i) => {
    const start = Math.floor(i * sentences.length / blockCount);
    const end = Math.floor((i + 1) * sentences.length / blockCount);
    return sentences.slice(start, end).join(" ");
  });
  const changes = blocks.slice(1).map((block, i) => ({ index: i + 1, score: 1 - cosine(vector(blocks[i]), vector(block)) }));
  const minGap = Math.max(1, Math.floor(blockCount / desired / 2));
  const selected: number[] = [];
  for (const point of changes.sort((a, b) => b.score - a.score)) {
    if (selected.every((value) => Math.abs(value - point.index) >= minGap)) selected.push(point.index);
    if (selected.length >= Math.min(desired - 1, blockCount - 1)) break;
  }
  const bounds = [0, ...selected.sort((a, b) => a - b), blockCount];
  return bounds.slice(0, -1).map((startBlock, i) => {
    const endBlock = bounds[i + 1];
    const chapterText = blocks.slice(startBlock, endBlock).join(" ");
    return {
      start: Math.round(startBlock / blockCount * duration),
      end: Math.round(endBlock / blockCount * duration),
      title: titleFor(chapterText),
      summary: splitSentences(chapterText)[0]?.slice(0, 180) || "Фрагмент выпуска"
    };
  });
}

export function youtubeDescription(chapters: Chapter[]) {
  return chapters.map((chapter) => `${formatTime(chapter.start)} ${chapter.title}`).join("\n");
}
