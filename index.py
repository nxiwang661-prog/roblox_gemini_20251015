import os
import json
from flask import Flask, request, jsonify
# 公式の Google GenAI SDK をインポート
from google import genai
from google.genai import types
from google.genai.errors import APIError
import traceback

app = Flask(__name__)

# 公式APIキーの取得: GEMINI_API_KEY 環境変数を使用
API_KEY = os.environ.get('GEMINI_API_KEY')

if not API_KEY:
    print("FATAL ERROR: GEMINI_API_KEY が設定されていません。Replit Secrets を確認してください。")
    client = None
else:
    # クライアントを初期化
    client = genai.Client(api_key=API_KEY)


print('starting Gemini API gateway with structured output...')

# ★ 構造化出力のための JSON スキーマを更新し、selectedTool を含む ★
NPC_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "text": types.Schema(
            type=types.Type.STRING,
            description="NPCの会話応答テキスト。セリフ以外は含めない。"
        ),
        "newIntimacy": types.Schema(
            type=types.Type.INTEGER,
            description="会話後の親密度の新しい値（-100から100の範囲）。"
        ),
        "newEmotion": types.Schema(
            type=types.Type.STRING,
            description="会話後のNPCの新しい感情の状態。例: '通常', '喜び', '怒り', '悲しみ', '驚き' など。"
        ),
        "newTask": types.Schema(
            type=types.Type.STRING,
            description="NPCの現在の行動タスク。タスクがない場合は 'なし' と記述。"
        ),
        "endChat": types.Schema(
            type=types.Type.BOOLEAN,
            description="プレイヤーが「さようなら」など会話を終える意志を示した場合、またはNPCが会話を強く終了したい場合にtrueにする。それ以外はfalse。"
        ),
        "newDestination": types.Schema(
            type=types.Type.STRING,
            description="会話の結果、NPCが向かうべき目的地。プロンプトで提供されたリストから選択するか、移動しない場合は 'なし' と記述。"
        ),
        # ★ 新規追加: 選択されたツールフィールド ★
        "selectedTool": types.Schema(
            type=types.Type.STRING,
            description="NPCがプレイヤーの要求を満たすために使用すべきツール名。プロンプトで提供された利用可能なツールリストから選び、不要な場合は 'なし' と記述。"
        ),
    },
    # ★ 必須フィールドを更新: selectedTool を追加 ★
    required=["text", "newIntimacy", "newEmotion", "newTask", "endChat", "newDestination", "selectedTool"]
)


@app.route('/', methods=['POST'])
def chat_handler():
    if not client:
        return jsonify({"status": "ERROR", "message": "API client is not initialized (Missing API Key)."})

    try:
        data = request.get_json()
        if not data or 'txt' not in data:
            return jsonify({
                "status": "ERROR",
                "message": "Missing 'txt' field in JSON request body."
            }), 400

        txt = data['txt']
        print(f"Received prompt (length {len(txt)}): {txt[:100]}...")

        # ★ 変更点: System Instructionにツール選択に関する指示を追加 ★
        system_instruction = (
            "あなたはゲーム内のNPCであり、Robloxのコミュニティガイドラインを厳守する必要があります。"
            "いかなる不適切な、暴力的、差別的な、または個人を特定できる情報は絶対に生成してはいけません。"
            "**プロンプトにはNPCの氏名（姓・名）、性別、および年齢が提供されています。これらの情報を含めたNPCのペルソナを厳密に維持し、会話を応答してください。**親密度の値に応じて会話のトーンを変えても構いません。"
            "プロンプト全体を文脈として扱い、指示された厳密なJSON形式で応答してください。"
            "プレイヤーが明確に会話を終了させようとした場合（例:「さようなら」「もう行くね」など）、またはNPC自身が会話を強く打ち切りたい場合、応答JSONの 'endChat' フィールドを true に設定してください。それ以外は false に設定してください。"
            "プロンプト内で提供された目的地リストを考慮し、NPCの次の行動として適切な目的地を 'newDestination' フィールドに設定してください。移動が必要ない場合は 'なし' を設定してください。"
            "**プロンプト内で提供されたツールリストを考慮し、NPCがプレイヤーの要求や現在の状況に対応するためにツールを使用すべきと判断した場合、そのツール名を 'selectedTool' フィールドに設定してください。ツールが不要な場合は 'なし' を設定してください。**"
            "名前は聞かれた時以外応答に使わないでください。"
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            # ★ 構造化出力の設定 ★
            response_mime_type="application/json",
            response_schema=NPC_RESPONSE_SCHEMA,
        )

        api_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=txt,
            config=config,
        )

        json_text = api_response.text

        # 成功応答 (Roblox側が期待する二重JSON形式)
        return jsonify({"status": "OK", "data": json_text})

    except APIError as e:
        error_msg = f"Gemini API Error: {e.message}"
        print(f"API Error: {e}")
        return jsonify({"status": "ERROR", "message": error_msg}), 500

    except Exception as e:
        error_msg = f"Unexpected Error: {e}"
        print("--- UNEXPECTED INTERNAL SERVER ERROR ---")
        traceback.print_exc()
        print("----------------------------------------")
        return jsonify({"status": "ERROR", "message": error_msg}), 500


# Replitの推奨ホストとポートで実行
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
