from multiapp import MultiApp
from apps import app_ref, app_tools, app_projs

app = MultiApp()

# Add all your application here
app.add_app("📚 參考資料", app_ref.app)
app.add_app("🧰 開發工具", app_tools.app)
app.add_app("🗃️ 其它專案", app_projs.app)

# The main app
app.run()
