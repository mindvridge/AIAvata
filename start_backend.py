#!/usr/bin/env python3
"""
백엔드 서버 시작 스크립트
"""
import sys
import os
import uvicorn

# 🔑 출력 버퍼링 비활성화 (즉시 출력)
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

if __name__ == "__main__":
    print("=" * 60)
    print("백엔드 서버 시작 중...")
    print("=" * 60)
    print("")
    
    try:
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True,
            use_colors=True,  # 🔑 컬러 출력 활성화
        )
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1)

