/**
 * 비디오 파일 캐싱 유틸리티
 * IndexedDB를 사용하여 비디오 파일을 브라우저에 캐싱
 */

const DB_NAME = 'avatar-video-cache';
const DB_VERSION = 1;
const STORE_NAME = 'videos';

interface CachedVideo {
  url: string;
  blob: Blob;
  timestamp: number;
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
        db.createObjectStore(STORE_NAME, { keyPath: 'url' });
      }
    };
  });
}

/**
 * 비디오 파일을 IndexedDB에 캐싱
 */
export async function cacheVideo(url: string): Promise<Blob> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);

    // 이미 캐시에 있는지 확인
    const cached = await new Promise<CachedVideo | null>((resolve, reject) => {
      const request = store.get(url);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });

    if (cached) {
      console.log(`%c📦 캐시에서 비디오 로드: ${url}`, 'color: blue; font-weight: bold');
      return cached.blob;
    }

    // 네트워크에서 다운로드
    console.log(`%c⬇️ 비디오 다운로드 중: ${url}`, 'color: orange; font-weight: bold');
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch video: ${response.statusText}`);
    }

    const blob = await response.blob();

    // IndexedDB에 저장
    const cacheEntry: CachedVideo = {
      url,
      blob,
      timestamp: Date.now(),
    };

    await new Promise<void>((resolve, reject) => {
      const request = store.put(cacheEntry);
      request.onsuccess = () => {
        console.log(`%c✅ 비디오 캐시 완료: ${url}`, 'color: green; font-weight: bold');
        resolve();
      };
      request.onerror = () => reject(request.error);
    });

    return blob;
  } catch (error) {
    console.error(`%c❌ 비디오 캐싱 실패: ${url}`, 'color: red; font-weight: bold', error);
    // 실패 시 네트워크 요청 시도
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch video: ${response.statusText}`);
    }
    return await response.blob();
  }
}

/**
 * 캐시된 비디오 URL 가져오기 (Blob URL 생성)
 */
export async function getCachedVideoURL(url: string): Promise<string> {
  const blob = await cacheVideo(url);
  return URL.createObjectURL(blob);
}

/**
 * 캐시 초기화 (선택적)
 */
export async function clearVideoCache(): Promise<void> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readwrite');
    const store = transaction.objectStore(STORE_NAME);
    await new Promise<void>((resolve, reject) => {
      const request = store.clear();
      request.onsuccess = () => {
        console.log('%c🗑️ 비디오 캐시 초기화 완료', 'color: orange; font-weight: bold');
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
export async function getCacheInfo(): Promise<{ count: number; size: number }> {
  try {
    const db = await openDB();
    const transaction = db.transaction(STORE_NAME, 'readonly');
    const store = transaction.objectStore(STORE_NAME);
    
    return new Promise((resolve, reject) => {
      const request = store.getAll();
      request.onsuccess = () => {
        const videos = request.result as CachedVideo[];
        const count = videos.length;
        const size = videos.reduce((sum, video) => sum + video.blob.size, 0);
        resolve({ count, size });
      };
      request.onerror = () => reject(request.error);
    });
  } catch (error) {
    console.error('%c❌ 캐시 정보 조회 실패', 'color: red; font-weight: bold', error);
    return { count: 0, size: 0 };
  }
}

