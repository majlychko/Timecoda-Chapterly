"use client";

import { useState } from "react";
import { Check, Clipboard, Clock3, FileText, Link2, Loader2, Sparkles, Upload, WandSparkles, Youtube } from "lucide-react";
import { Chapter, formatTime } from "../lib/chapters";

const DEMO = `Сегодня говорим о том, как подкасты меняют привычки аудитории. Сначала разберём рост рынка и почему длинные разговорные форматы снова стали популярны. Люди слушают выпуски в дороге, на тренировке и во время работы. Для брендов это редкий шанс получить целый час внимания. Затем перейдём к производству. Хороший выпуск начинается не с микрофона, а с ясной темы и сценарного каркаса. Продюсер заранее формулирует конфликт, готовит вопросы и определяет поворотные точки беседы. После записи команда получает длинный аудиофайл. Его нужно расшифровать, очистить и превратить в понятную структуру. Здесь появляется автоматическая разметка на главы. Алгоритм сравнивает соседние фрагменты текста и замечает смену темы. Затем редактор проверяет границы и уточняет заголовки. Отдельно поговорим о дистрибуции. YouTube использует таймкоды для навигации и показывает главы прямо в поиске. Короткие названия помогают зрителю быстро найти нужный фрагмент. В финале обсудим экономику. Автоматизация не заменяет продюсера, но убирает механическую работу. Команда быстрее выпускает эпизод, а редактор концентрируется на смысле и качестве.`;

export default function Home() {
  const [sourceType, setSourceType] = useState<"file" | "youtube" | "text">("file");
  const [mediaName, setMediaName] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [transcript, setTranscript] = useState(DEMO);
  const [duration, setDuration] = useState(48);
  const [count, setCount] = useState(5);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [description, setDescription] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [youtubeFallback, setYoutubeFallback] = useState(false);

  async function generate() {
    setLoading(true); setError(""); setCopied(false);
    try {
      if (sourceType === "youtube" && !youtubeUrl.trim()) throw new Error("Вставьте ссылку на YouTube-видео.");
      const useYoutubeApi = sourceType === "youtube" && !youtubeFallback;
      if (sourceType === "youtube" && youtubeFallback && !transcript.trim()) throw new Error("Вставьте расшифровку видео.");
      const endpoint = useYoutubeApi ? "/api/youtube" : "/api/chapters";
      const payload = useYoutubeApi
        ? { url: youtubeUrl, chapters: count }
        : { transcript, durationMinutes: duration, chapters: count };
      const response = await fetch(endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const data = await response.json();
      if (!response.ok) {
        if (sourceType === "youtube" && endpoint === "/api/youtube") {
          setYoutubeFallback(true);
          setTranscript("");
          throw new Error("YouTube ограничил автоматическое получение субтитров. Вставьте расшифровку ниже и повторите анализ.");
        }
        throw new Error(data.error);
      }
      if (data.transcript) setTranscript(data.transcript);
      if (data.durationMinutes) setDuration(data.durationMinutes);
      setChapters(data.chapters); setDescription(data.description);
    } catch (e) { setError(e instanceof Error ? e.message : "Ошибка"); }
    finally { setLoading(false); }
  }

  async function copy() {
    await navigator.clipboard.writeText(description); setCopied(true); setTimeout(() => setCopied(false), 1800);
  }

  return <main>
    <nav><div className="brand"><span><WandSparkles size={18}/></span> chapterly</div></nav>
    <header>
      <div>
        <h1>Главы для подкаста<br/><em>за пару минут.</em></h1>
      </div>
      <p>Загрузите аудио или видео, добавьте ссылку YouTube либо вставьте текст — сервис соберёт главы и таймкоды.</p>
    </header>
    <section className="journey" aria-label="Как это работает">
      <div><span>1</span><Upload size={18}/><p><b>Добавьте выпуск</b><small>Файл, YouTube или текст</small></p></div>
      <div><span>2</span><Clock3 size={18}/><p><b>Укажите длительность</b><small>И желаемое число глав</small></p></div>
      <div><span>3</span><Clipboard size={18}/><p><b>Скопируйте результат</b><small>Таймкоды готовы для YouTube</small></p></div>
    </section>
    <section className="workspace">
      <div className="inputPanel panel">
        <div className="panelTitle"><span>1</span><div><b>Добавьте выпуск</b><small>Выберите удобный источник</small></div></div>
        <div className="sourceTabs">
          <button className={sourceType === "file" ? "active" : ""} onClick={() => setSourceType("file")}><Upload size={15}/> Файл</button>
          <button className={sourceType === "youtube" ? "active" : ""} onClick={() => { setSourceType("youtube"); setYoutubeFallback(false); setError(""); }}><Youtube size={16}/> YouTube</button>
          <button className={sourceType === "text" ? "active" : ""} onClick={() => setSourceType("text")}><FileText size={15}/> Текст</button>
        </div>
        {sourceType === "file" && <label className="dropzone">
          <input type="file" accept="audio/*,video/*,.mp3,.wav,.m4a,.mp4,.mov,.webm" onChange={(e) => setMediaName(e.target.files?.[0]?.name || "")}/>
          <span><Upload size={22}/></span>
          <b>{mediaName || "Перетащите аудио или видео"}</b>
          <small>{mediaName ? "Файл выбран" : "или нажмите, чтобы выбрать · MP3, WAV, MP4, MOV"}</small>
        </label>}
        {sourceType === "youtube" && <div className="urlField"><Link2 size={18}/><input value={youtubeUrl} onChange={(e) => setYoutubeUrl(e.target.value)} placeholder="https://youtube.com/watch?v=…"/></div>}
        {sourceType === "file" && <div className="mvpHint"><Sparkles size={14}/><span><b>Режим MVP:</b> после выбора файла вставьте расшифровку ниже. Автотранскрибация подключается отдельным GPU‑воркером.</span></div>}
        {sourceType === "youtube" && !youtubeFallback && <div className="mvpHint ready"><Youtube size={14}/><span>Попробуем получить русские субтитры видео автоматически.</span></div>}
        {(sourceType !== "youtube" || youtubeFallback) && <label className="transcriptLabel"><span>{sourceType === "text" ? "Расшифровка выпуска" : sourceType === "youtube" ? "Расшифровка видео" : "Расшифровка для анализа"}</span><textarea value={transcript} onChange={(e) => setTranscript(e.target.value)} placeholder="Вставьте текст расшифровки…"/></label>}
        <div className="controls">
          <label><span>Длительность</span><div><input type="number" min="3" max="600" value={duration} onChange={(e) => setDuration(+e.target.value)}/><small>мин</small></div></label>
          <label><span>Количество глав</span><div><input type="number" min="3" max="15" value={count} onChange={(e) => setCount(+e.target.value)}/><small>глав</small></div></label>
        </div>
        <button className="primary" onClick={generate} disabled={loading}>{loading ? <><Loader2 className="spin" size={18}/> Анализируем темы…</> : <><Sparkles size={18}/> Собрать главы</>}</button>
        {error && <div className="error">{error}</div>}
        <p className="privacy">Текст обрабатывается без сохранения на сервере</p>
      </div>
      <div className="outputPanel panel">
        <div className="panelTitle"><span>2</span><div><b>Проверьте готовые главы</b><small>{chapters.length ? `${chapters.length} глав · можно поправить заголовки` : "Нажмите «Собрать главы» слева"}</small></div></div>
        {!chapters.length ? <div className="empty"><div><Clock3 size={30}/></div><b>Здесь появится результат</b><p>Мы покажем главы, таймкоды и готовый блок для описания видео.</p></div> : <>
          <div className="chapterList">{chapters.map((chapter, i) => <article key={chapter.start}><time>{formatTime(chapter.start)}</time><div><b contentEditable suppressContentEditableWarning>{chapter.title}</b><p>{chapter.summary}</p></div><span>{String(i + 1).padStart(2, "0")}</span></article>)}</div>
          <div className="export"><div><b>Описание для YouTube</b><small>таймкоды распознаются автоматически</small></div><pre>{description}</pre><button onClick={copy}>{copied ? <Check size={16}/> : <Clipboard size={16}/>} {copied ? "Скопировано" : "Копировать"}</button></div>
        </>}
      </div>
    </section>
    <footer><span>Основано на пайплайне GigaAM + USER2 + Qwen</span><span>Лёгкий TF-IDF режим для MVP</span></footer>
  </main>;
}
