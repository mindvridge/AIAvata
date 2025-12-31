/**
 * Idle 비디오 녹화 유틸리티
 * 프레임들을 수집하여 실제 비디오 파일로 변환
 */

/**
 * JPEG 프레임에서 이미지 크기 자동 감지
 * @returns Image 객체와 크기 정보 (재사용 가능)
 */
async function loadImageWithDimensions(frameData: ArrayBuffer): Promise<{
  img: HTMLImageElement;
  width: number;
  height: number;
  url: string;
}> {
  return new Promise((resolve, reject) => {
    const blob = new Blob([frameData], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    const img = new Image();

    img.onload = () => {
      console.log(`%c📐 이미지 크기 자동 감지: ${img.naturalWidth}x${img.naturalHeight}`, 'color: blue; font-weight: bold');
      resolve({
        img,
        width: img.naturalWidth,
        height: img.naturalHeight,
        url  // URL은 호출자가 revoke해야 함
      });
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to load image for dimension detection'));
    };

    img.src = url;
  });
}

/**
 * 프레임들을 비디오 Blob으로 변환
 * width/height가 제공되지 않으면 첫 번째 프레임에서 자동 감지
 */
export async function createVideoFromFrames(
  frames: ArrayBuffer[],
  width?: number,
  height?: number,
  fps: number = 30
): Promise<Blob> {
  if (frames.length === 0) {
    throw new Error('No frames to convert');
  }

  // 첫 번째 프레임을 로드하여 크기 감지 및 이미지 재사용
  let firstFrameInfo: { img: HTMLImageElement; width: number; height: number; url: string } | null = null;

  // 크기가 둘 다 제공되지 않은 경우만 자동 감지
  // 하나라도 제공되지 않으면 자동 감지된 크기 사용
  let finalWidth: number;
  let finalHeight: number;

  if (width !== undefined && height !== undefined) {
    // 둘 다 제공됨 - 제공된 값 사용
    finalWidth = width;
    finalHeight = height;
  } else {
    // 하나라도 없으면 자동 감지
    try {
      firstFrameInfo = await loadImageWithDimensions(frames[0]);
      finalWidth = width ?? firstFrameInfo.width;
      finalHeight = height ?? firstFrameInfo.height;
    } catch (error) {
      console.error('Failed to detect image dimensions, using fallback 784x1176:', error);
      finalWidth = width ?? 784;
      finalHeight = height ?? 1176;
    }
  }

  return new Promise((resolve, reject) => {
    // Canvas 생성
    const canvas = document.createElement('canvas');
    canvas.width = finalWidth;
    canvas.height = finalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      if (firstFrameInfo) URL.revokeObjectURL(firstFrameInfo.url);
      reject(new Error('Failed to get canvas context'));
      return;
    }

    console.log(`%c🎬 비디오 생성 시작: ${finalWidth}x${finalHeight}, ${frames.length} frames, ${fps}fps`, 'color: orange; font-weight: bold');

    // Canvas에서 스트림 캡처
    const stream = canvas.captureStream(fps);

    // MediaRecorder 생성
    const mimeTypes = [
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
    ];

    let selectedMimeType = '';
    for (const mimeType of mimeTypes) {
      if (MediaRecorder.isTypeSupported(mimeType)) {
        selectedMimeType = mimeType;
        break;
      }
    }

    if (!selectedMimeType) {
      if (firstFrameInfo) URL.revokeObjectURL(firstFrameInfo.url);
      reject(new Error('WebM 형식을 지원하지 않는 브라우저입니다.'));
      return;
    }

    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: selectedMimeType,
      videoBitsPerSecond: 2000000, // 2Mbps
    });

    const chunks: Blob[] = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunks.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      if (chunks.length > 0) {
        const blob = new Blob(chunks, { type: selectedMimeType });
        console.log(`%c✅ 비디오 생성 완료: ${finalWidth}x${finalHeight}, ${(blob.size / 1024).toFixed(2)}KB`, 'color: green; font-weight: bold');
        resolve(blob);
      } else {
        reject(new Error('No video data recorded'));
      }
    };

    mediaRecorder.onerror = () => {
      reject(new Error('MediaRecorder error'));
    };

    // 녹화 시작
    mediaRecorder.start();

    // 프레임들을 순차적으로 canvas에 그리기
    let frameIndex = 0;
    const frameInterval = 1000 / fps; // ms per frame

    const drawNextFrame = () => {
      if (frameIndex >= frames.length) {
        // 모든 프레임을 그렸으면 녹화 중지
        setTimeout(() => {
          if (mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
          }
        }, frameInterval);
        return;
      }

      const frameData = frames[frameIndex];
      const blob = new Blob([frameData], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      const img = new Image();

      img.onload = () => {
        // Canvas에 프레임 그리기
        ctx.clearRect(0, 0, finalWidth, finalHeight);
        ctx.drawImage(img, 0, 0, finalWidth, finalHeight);
        URL.revokeObjectURL(url);

        frameIndex++;
        // 다음 프레임으로 (FPS에 맞춰)
        setTimeout(drawNextFrame, frameInterval);
      };

      img.onerror = () => {
        URL.revokeObjectURL(url);
        frameIndex++;
        setTimeout(drawNextFrame, frameInterval);
      };

      img.src = url;
    };

    // 첫 프레임 그리기 (이미 로드된 이미지가 있으면 재사용)
    if (firstFrameInfo) {
      // 이미 로드된 첫 프레임 재사용
      ctx.drawImage(firstFrameInfo.img, 0, 0, finalWidth, finalHeight);
      URL.revokeObjectURL(firstFrameInfo.url);
      frameIndex = 1;
      // 약간의 지연 후 다음 프레임 시작 (MediaRecorder가 초기화될 시간 확보)
      setTimeout(drawNextFrame, frameInterval);
    } else {
      // 첫 프레임을 새로 로드
      const firstFrameBlob = new Blob([frames[0]], { type: 'image/jpeg' });
      const firstFrameUrl = URL.createObjectURL(firstFrameBlob);
      const firstImg = new Image();

      firstImg.onload = () => {
        ctx.drawImage(firstImg, 0, 0, finalWidth, finalHeight);
        URL.revokeObjectURL(firstFrameUrl);
        frameIndex = 1;
        setTimeout(drawNextFrame, frameInterval);
      };

      firstImg.onerror = () => {
        URL.revokeObjectURL(firstFrameUrl);
        reject(new Error('Failed to load first frame'));
      };

      firstImg.src = firstFrameUrl;
    }
  });
}
