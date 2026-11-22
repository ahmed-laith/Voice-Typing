// VOX MOTION STYLE CONTROLLER WITH SYNCED HABIBI-TTS AUDIO

const scenes = [
  { id: 'scene1', start: 0, duration: 8.5, audioSrc: 'audio/01.wav' },
  { id: 'scene2', start: 8.5, duration: 7.5, audioSrc: 'audio/02.wav' },
  { id: 'scene3', start: 16.0, duration: 9.5, audioSrc: 'audio/03.wav' },
  { id: 'scene4', start: 25.5, duration: 8.5, audioSrc: 'audio/04.wav' },
  { id: 'scene5', start: 34.0, duration: 6.0, audioSrc: null }
];

const totalDuration = 40; // seconds
let currentTime = 0;
let isPlaying = false;
let isAudioEnabled = true;
let timer = null;
let currentSceneIndex = -1;

// Audio instances
const audioMap = {};
scenes.forEach(s => {
  if (s.audioSrc) {
    const aud = new Audio(s.audioSrc);
    aud.preload = 'auto';
    audioMap[s.id] = aud;
  }
});

const btnPlay = document.getElementById('btnPlay');
const btnReset = document.getElementById('btnReset');
const btnAudioToggle = document.getElementById('btnAudioToggle');
const timecode = document.getElementById('timecode');
const progressBar = document.getElementById('progressBar');
const typedTextEl = document.getElementById('typedText');

const demoSpeechText = "افتحلي Claude Code وسوي Commit للبروجكت وخلينا نشرب چاي...";

function formatTime(s) {
  const m = Math.floor(s / 60).toString().padStart(2, '0');
  const sec = Math.floor(s % 60).toString().padStart(2, '0');
  return `${m}:${sec}`;
}

function stopAllAudio() {
  Object.values(audioMap).forEach(aud => {
    aud.pause();
    aud.currentTime = 0;
  });
}

function syncAudio(time, activeIndex) {
  if (!isAudioEnabled) return;
  
  if (activeIndex === -1) {
    stopAllAudio();
    return;
  }

  const activeScene = scenes[activeIndex];
  const targetAudio = audioMap[activeScene.id];

  // Pause audios of other scenes
  scenes.forEach((s, idx) => {
    if (idx !== activeIndex && audioMap[s.id]) {
      audioMap[s.id].pause();
      audioMap[s.id].currentTime = 0;
    }
  });

  if (targetAudio && isPlaying) {
    const sceneOffset = time - activeScene.start;
    if (targetAudio.paused || Math.abs(targetAudio.currentTime - sceneOffset) > 0.4) {
      if (sceneOffset >= 0 && sceneOffset < (targetAudio.duration || 999)) {
        try {
          targetAudio.currentTime = sceneOffset;
          targetAudio.play().catch(e => console.log('Audio playback waiting for interaction:', e));
        } catch(e) {}
      }
    }
  }
}

function updateScene(time) {
  let activeIndex = -1;
  for (let i = 0; i < scenes.length; i++) {
    const s = scenes[i];
    if (time >= s.start && time < (s.start + s.duration)) {
      activeIndex = i;
      break;
    }
  }

  scenes.forEach((s, idx) => {
    const el = document.getElementById(s.id);
    if (idx === activeIndex) {
      el.classList.add('active');
    } else {
      el.classList.remove('active');
    }
  });

  // Update scene tab active classes
  document.querySelectorAll('.scene-tab').forEach((tab, idx) => {
    if (idx === activeIndex) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });

  // Audio Sync
  if (activeIndex !== currentSceneIndex) {
    currentSceneIndex = activeIndex;
    if (isPlaying) {
      syncAudio(time, activeIndex);
    }
  }

  // Typewriter effect in Scene 3
  if (activeIndex === 2) {
    const sceneProgress = (time - scenes[2].start) / scenes[2].duration;
    const charCount = Math.floor(sceneProgress * demoSpeechText.length);
    typedTextEl.textContent = demoSpeechText.slice(0, charCount);
  } else if (activeIndex < 2) {
    typedTextEl.textContent = "";
  } else {
    typedTextEl.textContent = demoSpeechText;
  }

  // Update progress bar & timecode
  const progressPercent = Math.min(100, (time / totalDuration) * 100);
  progressBar.style.width = `${progressPercent}%`;
  timecode.textContent = `${formatTime(time)} / ${formatTime(totalDuration)}`;
}

function play() {
  if (isPlaying) {
    pause();
    return;
  }
  isPlaying = true;
  btnPlay.textContent = "إيقاف مؤقت ⏸";
  btnPlay.style.background = "#FFCC00";
  btnPlay.style.color = "#000";

  syncAudio(currentTime, currentSceneIndex);

  const stepMs = 100;
  timer = setInterval(() => {
    currentTime += stepMs / 1000;
    if (currentTime >= totalDuration) {
      pause();
      currentTime = totalDuration;
      stopAllAudio();
    }
    updateScene(currentTime);
  }, stepMs);
}

function pause() {
  isPlaying = false;
  btnPlay.textContent = "تشغيل العرض ▶";
  btnPlay.style.background = "var(--accent-orange)";
  btnPlay.style.color = "#fff";
  if (timer) clearInterval(timer);
  Object.values(audioMap).forEach(aud => aud.pause());
}

function reset() {
  pause();
  stopAllAudio();
  currentTime = 0;
  currentSceneIndex = -1;
  updateScene(0);
}

function toggleAudio() {
  isAudioEnabled = !isAudioEnabled;
  if (!isAudioEnabled) {
    btnAudioToggle.textContent = "الصوت: كتم 🔇";
    btnAudioToggle.style.opacity = "0.7";
    stopAllAudio();
  } else {
    btnAudioToggle.textContent = "الصوت: مفعل 🔊";
    btnAudioToggle.style.opacity = "1";
    if (isPlaying) syncAudio(currentTime, currentSceneIndex);
  }
}

btnPlay.addEventListener('click', play);
btnReset.addEventListener('click', reset);
if (btnAudioToggle) btnAudioToggle.addEventListener('click', toggleAudio);

// Scene tabs navigation
document.querySelectorAll('.scene-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const idx = parseInt(tab.getAttribute('data-scene'), 10);
    if (idx >= 0 && idx < scenes.length) {
      currentTime = scenes[idx].start;
      updateScene(currentTime);
      if (isPlaying) {
        syncAudio(currentTime, idx);
      }
    }
  });
});

// Clickable progress bar to seek
const reelProgress = document.querySelector('.reel-progress');
if (reelProgress) {
  reelProgress.addEventListener('click', (e) => {
    const rect = reelProgress.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    currentTime = ratio * totalDuration;
    updateScene(currentTime);
    if (isPlaying) {
      syncAudio(currentTime, currentSceneIndex);
    }
  });
}

// Initial state / URL parameter jump
const urlParams = new URLSearchParams(window.location.search);
const targetSceneParam = urlParams.get('scene');
if (targetSceneParam) {
  const sceneNum = parseInt(targetSceneParam, 10);
  if (sceneNum >= 1 && sceneNum <= scenes.length) {
    const s = scenes[sceneNum - 1];
    currentTime = s.start + (sceneNum === 3 ? 5.0 : 1.0);
    updateScene(currentTime);
  } else {
    updateScene(0);
  }
} else {
  updateScene(0);
}
