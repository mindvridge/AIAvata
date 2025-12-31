/**
 * Idle 비디오 캐싱 유틸리티
 * 서버 스트림에서 받은 프레임들을 비디오 파일로 변환하여 캐시
 */

const DB_NAME = 'avatar-idle-video-cache';
const DB_VERSION = 1;
const STORE_NAME = 'idleVideos';

interface CachedIdleVideo {
  emotion: string; // 'neutral', 'happy' 등
  videoBlob: Blob; // 비디오 파일 (WebM 형식)
  timestamp: number;
  frameCount: number;
}

/**
 * IndexedDB 데이터베이스 열기
 */
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'emotion' });
        store.createIndex('timestamp', 'timestamp', { unique: false });
      }
    };
  });
}

/**
 * 캐시된 idle 비디오 가져오기
 */
export async function getCachedIdleVideo(emotion: string = 'neutral'): Promise<Blob | null> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.get(emotion);
      request.onsuccess = () => {
        const cached = request.result as CachedIdleVideo | undefined;
        if (cached && cached.videoBlob) {
          console.log(`%c📦 캐시에서 idle 비디오 로드: ${emotion} (${(cached.videoBlob.size / 1024).toFixed(2)}KB)`, 'color: blue; font-weight: bold');
          resolve(cached.videoBlob);
        } else {
          resolve(null);
        }
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error(`%c❌ 캐시 조회 실패: ${emotion}`, 'color: red; font-weight: bold', error);
    return null;
  }
}

/**
 * 프레임들로부터 idle 비디오 생성 및 캐시 저장
 * width/height가 제공되지 않으면 첫 번째 프레임에서 자동 감지
 */
export async function cacheIdleVideoFrames(
  frames: ArrayBuffer[],
  emotion: string = 'neutral',
  width?: number,  // 제공되지 않으면 첫 번째 프레임에서 자동 감지
  height?: number,  // 제공되지 않으면 첫 번째 프레임에서 자동 감지
  fps: number = 30
): Promise<void> {
  try {
    if (frames.length === 0) {
      console.warn(`%c⚠️ 프레임이 없어서 캐시하지 않음: ${emotion}`, 'color: orange; font-weight: bold');
      return;
    }

    console.log(`%c⬇️ Idle 비디오 생성 중: ${emotion} (${frames.length} frames, 크기: ${width || 'auto'}x${height || 'auto'})`, 'color: orange; font-weight: bold');

    // 프레임들을 비디오 Blob으로 변환 (크기 자동 감지 지원)
    const { createVideoFromFrames } = await import('./idleVideoRecorder');
    const videoBlob = await createVideoFromFrames(frames, width, height, fps);

    // IndexedDB에 저장
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    const cacheEntry: CachedIdleVideo = {
      emotion,
      videoBlob,
      timestamp: Date.now(),
      frameCount: frames.length,
    };

    await new Promise<void>((resolve, reject) => {
      const request = store.put(cacheEntry);
      request.onsuccess = () => {
        console.log(`%c✅ Idle 비디오 캐시 완료: ${emotion} (${(videoBlob.size / 1024).toFixed(2)}KB, ${frames.length} frames)`, 'color: green; font-weight: bold');
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error(`%c❌ Idle 비디오 캐싱 실패: ${emotion}`, 'color: red; font-weight: bold', error);
    throw error;
  }
}

/**
 * 캐시 초기화
 */
export async function clearIdleVideoCache(): Promise<void> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    await new Promise<void>((resolve, reject) => {
      const request = store.clear();
      request.onsuccess = () => {
        console.log('%c🗑️ Idle 비디오 캐시 초기화 완료', 'color: orange; font-weight: bold');
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('%c❌ 캐시 초기화 실패', 'color: red; font-weight: bold', error);
    throw error;
  }
}

/**
 * 캐시 정보 가져오기
 */
export async function getIdleVideoCacheInfo(): Promise<{ count: number; size: number; emotions: string[] }> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);

    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => {
        const videos = request.result as CachedIdleVideo[];
        const count = videos.length;
        const size = videos.reduce((sum, video) => sum + video.videoBlob.size, 0);
        const emotions = videos.map(v => v.emotion);
        resolve({ count, size, emotions });
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('%c❌ 캐시 정보 조회 실패', 'color: red; font-weight: bold', error);
    return { count: 0, size: 0, emotions: [] };
  }
}
