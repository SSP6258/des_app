from multiapp import MultiApp
from apps import app_ref, app_tools, app_projs, app_sim_1, app_hospital

app = MultiApp()

# Add all your application here
app.add_app("🛒 應用: 請支援收銀~", app_sim_1.app)
app.add_app("🏥 應用: 急診很忙~", app_hospital.app)
app.add_app("📚 參考資料", app_ref.app)
app.add_app("🧰 開發工具", app_tools.app)
app.add_app("🗃️ 其它專案", app_projs.app)

# The main app
app.run()
