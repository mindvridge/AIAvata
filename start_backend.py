#!/usr/bin/env python3
"""
백엔드 서버 시작 스크립트
"""
import sys
import uvicorn

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
        )
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        sys.exit(1)

