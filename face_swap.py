"""
Homework 1 - Face Swap (InsightFace)

組員:
  11127163  黃理貞
  11127260  羅尹熙
  11327225  郭宣翎

說明：
這支程式是把一張「來源人臉圖片」套到「一整段影片」裡面。
做法就是一幀一幀處理影片 → 偵測臉 → 確認是不是要換的那個人 → 換臉 → 合成回影片。
最後再把原本影片的音訊合回去。

【執行方式】
python face_swap.py --source_image D:\Homework1\Huang.JPG --input_video D:\Homework1\AI-lumama.mp4 --output_video D:\Homework1\11127163_lumama.mp4

【PowerShell 可以換行的寫法】
python .\face_swap.py `
  --source_image D:\Homework1\Huang.JPG `
  --input_video  D:\Homework1\AI-lumama.mp4 `
  --output_video D:\Homework1\11127163_lumama.mp4

備註：
1. 如果 face_swap.py 不在目前資料夾，要把路徑寫完整，例如 python D:\Homework1\face_swap.py ...
2. 如果有參考臉要過濾主角，可以多加：
   --reference_face D:\Homework1\main_face.jpg
3. 如果要開臉部增強，可加：
   --enhance_model GFPGAN
"""

import argparse          # 讀命令列參數用的
import os                # 處理路徑、判斷檔案在不在
import subprocess        # 等一下要呼叫 ffmpeg
import cv2               # OpenCV，讀影片、寫影片、影像處理
import numpy as np       # 做陣列計算比較方便

from insightface.app import FaceAnalysis   # InsightFace 的人臉偵測/特徵
from insightface.model_zoo import model_zoo  # 用來載 onnx 的換臉模型

# 這兩個是加強臉用的套件，不是一定要有
# try import，因為環境不一定有裝
try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
except ImportError:
    RealESRGANer = None


def get_ffmpeg_bin():
    """
    這個函式的目的：找到 ffmpeg 執行檔在哪裡。
    因為每個人電腦放的位置不同，所以依序去找幾個可能的位置。
    找不到的話就回傳 'ffmpeg'，讓系統自己去 PATH 裡找。
    """
    p = os.environ.get('FFMPEG_BIN')
    if p and os.path.exists(p):
        return p

    # 這幾個是可能會放的路徑，如果放別的地方可以自己改
    cand = [
        r"D:\ffmpeg\ffmpeg-8.0-full_build\bin\ffmpeg.exe",
        r"D:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "ffmpeg"
    ]
    for c in cand:
        if os.path.exists(c):
            return c
    return "ffmpeg"   # 真的沒找到就交給系統自己找



# ===== 小工具區 =====

def l2norm(v, eps=1e-9):
    """
    把向量除以它自己的長度，讓它的長度變成 1
    這樣在算相似度的時候，會比較準，不會被向量有多長影響
    """
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + eps
    return v / n


def cosine_sim(a, b):
    """
    算兩個人臉向量的 cos 相似度
    用在「影片這張臉」跟「參考照片的臉」是不是同一個人
    回傳值越接近 1 代表越像
    """
    if a is None or b is None:
        return -1.0
    return float(np.dot(l2norm(a), l2norm(b)))


def make_circular_mask(h, w, feather=0.20):
    """
    做一張「中間實心、邊緣漸層」的圓形遮罩 (mask)。
    為什麼要做這張？因為等一下要把臉貼回去的時候，
    用這種漸層邊界，貼起來會比較自然，不會有方框感。
    h, w 是這張臉的高和寬。
    feather 是邊緣要模糊的比例，0.2 就是 20%。
    """
    y, x = np.ogrid[:h, :w]
    cy, cx = h/2.0, w/2.0
    r = min(h, w)/2.0
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)

    # 先做一個純圓
    mask = (dist <= r).astype(np.float32)

    # 再把靠近邊的那一圈做漸層
    band = (dist > r*(1-feather)) & (dist <= r)
    mask[band] = (r - dist[band]) / (r * feather)

    return (mask * 255).astype(np.uint8)



# 這兩個變數是「全域的模型實體」
# 這樣做的原因是：模型很大，不要每幀載一次
FACE_DETECTOR = None
FACE_SWAPPER  = None



class FaceSwapper:
    """
    這個 class 把整個流程包起來
    建構子裡先記住使用者給的路徑，後面就可以直接用
    """
    def __init__(self, source_path, input_video_path, output_video_path,
                 reference_path=None, enhance_model='None',
                 swapper_model_path=None, show_preview=False):

        # 這三個是一定要給的
        self.source_path = source_path        # 要換的那張臉
        self.input_video_path = input_video_path  # 要處理的影片
        self.output_video_path = output_video_path  # 輸出的影片

        # 下面這些是「可選的」
        self.reference_path = reference_path      # 如果影片裡有很多人，這張用來指定要換誰
        self.enhance_model  = enhance_model       # 要不要做臉部增強
        self.swapper_model_path = swapper_model_path  # 換臉模型如果不在預設位置可以自己指定
        self.show_preview   = show_preview        # 要不要一邊處理一邊看畫面

        # 先把無音訊的影片寫到這個暫存檔，最後再合音
        self.temp_video_path = "temp_video_no_audio.mp4"

        # 這兩個要等載完模型才能算
        self.reference_face_embedding = None
        self.source_face = None

        # 增強用的物件，先設為 None，有載到再放
        self.gfpganer = None
        self.sr = None


    # -------------- 載模型 --------------
    def load_models(self):
        """
        1. 載 InsightFace 的臉偵測/特徵
        2. 載 InSwapper 的換臉模型
        3. 看使用者有沒有指定要開臉部增強
        4. 把來源臉 & (可能有的) 參考臉先算好
        """
        global FACE_DETECTOR, FACE_SWAPPER

        print("載入模型中...")

        # 列兩個 provider，GPU 沒有的話就會用 CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # (1) 臉偵測 + 臉特徵
        FACE_DETECTOR = FaceAnalysis(
            name='buffalo_l',
            providers=providers,
            allowed_modules=['detection', 'recognition']
        )
        # ctx_id=0 表示想用 GPU，沒有就會回到 CPU
        FACE_DETECTOR.prepare(ctx_id=0, det_size=(640, 640))
        print("臉分析載入完成")

        # (2) 換臉模型
        if self.swapper_model_path:
            swapper_path = self.swapper_model_path
        else:
            # 路徑
            swapper_path = r"D:\Homework1\models\inswapper_128.onnx"

        if not os.path.exists(swapper_path):
            # 直接丟錯，不然後面一定會炸
            raise FileNotFoundError(f"找不到換臉模型：{swapper_path}")

        FACE_SWAPPER = model_zoo.get_model(swapper_path, providers=providers)
        if FACE_SWAPPER is None:
            raise RuntimeError("InSwapper 載入失敗")
        print("換臉模型載入完成")

        # (3) 看要不要開臉部增強
        # 「有就做，沒就算了」，是為了讓程式在沒有外部模型時也能跑完
        if self.enhance_model == 'GFPGAN' and GFPGANer:
            try:
                self.gfpganer = GFPGANer(
                    model_path='GFPGANv1.4.pth',  # 要放實際模型檔
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None
                )
                print("GFPGAN 已啟用")
            except Exception as e:
                print("GFPGAN 載入失敗：", e)
                self.gfpganer = None

        elif self.enhance_model == 'Real-ESRGAN' and RealESRGANer:
            try:
                model = SRVGGNetCompact(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_conv=32, upscale=2, act_type='prelu'
                )
                self.sr = RealESRGANer(
                    scale=2,
                    model_path='realesr-general-x2v3.pth',  # 一樣要自己放
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False
                )
                print("Real-ESRGAN 已啟用")
            except Exception as e:
                print("Real-ESRGAN 載入失敗：", e)
                self.sr = None
        else:
            print("這次不使用臉部增強")

        # (4) 先把來源臉找到，等等影片裡的臉都要換成這張
        src_img = cv2.imread(self.source_path)
        if src_img is None:
            raise FileNotFoundError(f"讀不到來源影像：{self.source_path}")

        src_faces = FACE_DETECTOR.get(src_img)
        if not src_faces:
            raise FileNotFoundError(f"來源影像沒有臉：{self.source_path}")

        # 如果來源圖裡有很多臉，就挑最大那張
        self.source_face = max(
            src_faces,
            key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])
        )

        # (5) 有給參考臉，就把他的人臉向量存起來，之後拿來比對是不是同一個人
        self.reference_face_embedding = None
        if self.reference_path:
            ref_img = cv2.imread(self.reference_path)
            if ref_img is None:
                raise FileNotFoundError(f"讀不到參考影像：{self.reference_path}")

            ref_faces = FACE_DETECTOR.get(ref_img)
            if not ref_faces:
                raise FileNotFoundError(f"參考影像沒有臉：{self.reference_path}")

            ref_main = max(
                ref_faces,
                key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])
            )
            self.reference_face_embedding = np.array(ref_main.normed_embedding, dtype=np.float32)
            print("參考臉已建立向量")  # 之後影片裡的臉會跟這個比對



    def calculate_similarity(self, embed1, embed2):
        """包一層，讓程式看起來比較好讀"""
        return cosine_sim(embed1, embed2)



    def _maybe_enhance_patch(self, face_patch):
        """
        face_patch 是一小塊臉 (從換好臉的圖裡切出來的)
        有載到增強模型就加強一下，沒有就直接回傳
        這樣即使沒有下載增強模型，程式也能正常跑完
        """
        enhanced = face_patch

        if self.gfpganer is not None:
            try:
                _, _, restored = self.gfpganer.enhance(
                    face_patch,
                    has_aligned=False,
                    only_center_face=True,
                    paste_back=False
                )
                if restored is not None:
                    enhanced = restored
            except Exception as e:
                print("GFPGAN 增強失敗：", e)

        elif self.sr is not None:
            try:
                enhanced, _ = self.sr.enhance(face_patch, outscale=1)
            except Exception as e:
                print("Real-ESRGAN 增強失敗：", e)

        return enhanced



    def swap_and_enhance(self, frame, target_face, source_face):
        """
        真的對一張臉動手
        步驟：
          1. 用 InSwapper 把 target_face 換成 source_face
          2. 把那塊臉切出來
          3. 有需要就增強
          4. 用無縫貼回 (seamlessClone)
        """
        # 先用 insightface 的換臉功能，直接換並貼在 frame 上
        swapped = FACE_SWAPPER.get(frame, target_face, source_face, paste_back=True)

        # 把這張臉所在的方框抓出來，等等要從這裡取圖
        x1, y1, x2, y2 = map(int, target_face.bbox.astype(int))
        # 安全檢查，避免超出畫面
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(swapped.shape[1]-1, x2), min(swapped.shape[0]-1, y2)

        face_patch = swapped[y1:y2, x1:x2].copy()
        h, w = face_patch.shape[:2]
        if h <= 0 or w <= 0:
            # 偶爾 bbox 會怪怪的，這時就直接回原圖
            return swapped

        # 有載增強模型就處理一下
        enhanced_patch = self._maybe_enhance_patch(face_patch)

        # 增強完如果大小跑掉，調回原大小
        if enhanced_patch.shape[:2] != (h, w):
            enhanced_patch = cv2.resize(enhanced_patch, (w, h))

        # 做一張圓形 mask，貼回去比較自然
        mask = make_circular_mask(h, w, feather=0.20)
        center = ((x1+x2)//2, (y1+y2)//2)

        blended = cv2.seamlessClone(
            enhanced_patch, swapped, mask, center, cv2.NORMAL_CLONE
        )
        return blended



    # --------------- 主流程：處理影片 ----------------
    def process_video(self):
        """
        主要流程：
        1. 把模型都載好
        2. 打開影片，一幀一幀讀
        3. 每一幀找臉、過濾、換臉
        4. 寫成一個「沒有聲音」的影片
        5. 換完之後再去合音
        """
        self.load_models()
        print("開始處理影片：", self.input_video_path)

        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"無法開啟影片：{self.input_video_path}")

        # 先把輸出影片要用的資訊取出來
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-2:
            # 有些影片讀不到 fps，就給一個常見的
            fps = 25.0

        # 建立一個沒有音訊的輸出影片 (先暫存)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.temp_video_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("無法建立輸出影片")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = frame.copy()
            faces = FACE_DETECTOR.get(frame)

            for tgt in faces:
                # 先假設要換
                do_swap = True

                # 如果有指定參考臉，就要比對是不是同一個人
                if self.reference_face_embedding is not None:
                    cur_emb = np.array(tgt.normed_embedding, dtype=np.float32)
                    sim = self.calculate_similarity(cur_emb, self.reference_face_embedding)
                    # 0.30 自己設的門檻，不到就不換
                    if sim < 0.30:
                        do_swap = False

                if do_swap:
                    processed = self.swap_and_enhance(processed, tgt, self.source_face)

            writer.write(processed)
            frame_idx += 1

            # 偶爾印一下幀數，知道還在跑
            if frame_idx % 50 == 0:
                print(f"已處理 {frame_idx} 幀")

            # 如果有加 --show，就順便顯示
            if self.show_preview:
                cv2.imshow("FaceSwap Preview", processed)
                # 按 ESC 可以中斷
                if cv2.waitKey(1) & 0xFF == 27:
                    print("使用者中止預覽")
                    break

        cap.release()
        writer.release()
        if self.show_preview:
            cv2.destroyAllWindows()

        print("影片逐幀處理完成（目前還沒合音）")
        self.re_synthesize_audio_with_ffmpeg()

        # 用不到的暫存檔刪掉
        if os.path.exists(self.temp_video_path) and \
           os.path.abspath(self.temp_video_path) != os.path.abspath(self.output_video_path):
            try:
                os.remove(self.temp_video_path)
            except Exception:
                pass



    # ---------- 合成音訊 ----------
    def re_synthesize_audio_with_ffmpeg(self):
        """
        把原本影片的音訊抽出來，再跟處理後的影片合在一起
        有音訊就合，沒音訊就只留畫面
        """
        print("開始用 FFmpeg 合成音訊...")

        temp_audio_path = "temp_audio.aac"
        ff = get_ffmpeg_bin()

        # 1. 先把舊影片的音訊抽出來
        extract_cmd = [
            ff, "-y",
            "-i", self.input_video_path,
            "-vn", "-acodec", "copy",
            temp_audio_path
        ]

        have_audio = True
        try:
            subprocess.run(extract_cmd, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("音訊抽取完成")
        except Exception:
            have_audio = False
            print("原影片可能沒有音訊，會輸出無音訊影片")

        # 2. 再把換完臉的影片 + 音訊 合成新的影片
        if have_audio:
            encode_cmd = [
                ff, "-y",
                "-i", self.temp_video_path,
                "-i", temp_audio_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "192k",
                self.output_video_path
            ]
        else:
            # 沒有音訊的情況
            encode_cmd = [
                ff, "-y",
                "-i", self.temp_video_path,
                "-map", "0:v:0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                self.output_video_path
            ]

        try:
            subprocess.run(encode_cmd, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("輸出完成：", self.output_video_path)
        except subprocess.CalledProcessError:
            # 如果合不起來，就至少給一個無音訊版本
            print("FFmpeg 合成失敗，改存無音訊版本")
            try:
                if os.path.exists(self.output_video_path):
                    os.remove(self.output_video_path)
                os.replace(self.temp_video_path, self.output_video_path)
            except Exception:
                pass
        finally:
            # 把暫存的音訊刪掉，乾淨一點
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except Exception:
                    pass



# ----------------- main -----------------
def main():
    """
    1. 把使用者在命令列打的參數讀進來
    2. 建立 FaceSwapper 物件
    3. 開始跑
    """
    parser = argparse.ArgumentParser(
        description="Face-Swap (InsightFace) | CYCU CSE",
        epilog="範例: python face_swap.py --source_image src.jpg --input_video in.mp4 --output_video out.mp4"
    )
    parser.add_argument('-source', '--source_image', required=True,
                        help='要拿來換的來源人臉圖片路徑')
    parser.add_argument('-input', '--input_video', required=True,
                        help='要處理的影片路徑')
    parser.add_argument('-output', '--output_video', required=True,
                        help='輸出的影片路徑')
    parser.add_argument('-ref', '--reference_face', default=None,
                        help='影片裡主角的正面照，拿來比對是不是同一個人 (相似度 < 0.30 就不換)')
    parser.add_argument('-enhance', '--enhance_model', default='None',
                        choices=['GFPGAN', 'Real-ESRGAN', 'None'],
                        help='要不要做臉部增強（可省略）')
    parser.add_argument('--swapper_model', default=None,
                        help='inswapper_128.onnx 的路徑，沒給就用程式內寫死的')
    parser.add_argument('--show', action='store_true',
                        help='處理時顯示畫面，ESC 可以中止')

    args = parser.parse_args()

    try:
        fs = FaceSwapper(
            source_path=args.source_image,
            input_video_path=args.input_video,
            output_video_path=args.output_video,
            reference_path=args.reference_face,
            enhance_model=args.enhance_model,
            swapper_model_path=args.swapper_model,
            show_preview=args.show
        )
        fs.process_video()
    except FileNotFoundError as e:
        # 檔案路徑錯誤
        print("錯誤：", e)
    except Exception as e:
        # 其他沒想過的錯誤
        print("發生未預期錯誤：", e)



if __name__ == '__main__':
    main()
