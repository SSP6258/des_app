## 概述：
這份文件是針對一個使用 Python 的 simpy 庫建立的離散事件模擬器進行分析和摘要。  
該模擬器旨在模擬一個超市收銀的情境，以了解顧客抵達、排隊和收銀過程中的行為和指標。  
程式碼使用 streamlit 框架建立了一個互動式的網頁介面，讓使用者可以調整模擬參數並視覺化結果。  

## 主要主題與重要概念：
1. 離散事件模擬 (Discrete Event Simulation, DES)：  
    - 程式碼核心是基於 DES 的概念，使用 simpy 庫來管理事件和時間。
    - DES 的優點在於能夠精確地模擬系統中事件發生的順序和時間，從而分析系統的效能。
    - 程式碼定義了兩種主要的模擬「程序」（process）：fn_sim_customer (顧客抵達) 和 fn_sim_cashier (收銀員服務)。
      
2. 收銀情境模擬：
    - 模擬了一個超市（全聯福利中心）的收銀排隊場景。
    - 目的是回答「何時需要廣播『請支援收銀』？」這個實際問題，即分析在不同參數下，排隊情況的嚴重程度。
      
3. 可配置的模擬參數：
    - 程式碼允許使用者透過 streamlit 介面調整關鍵參數，以探索不同情境下的模擬結果：
      * CUSTOMER_NUM (顧客數量)：模擬的顧客總數。
      * CASHIER_NUM (收銀員數量)：可用的收銀員數量。
      * CASHIER_TIME (收銀時間)：每位顧客完成收銀所需的時間（分鐘）。
      * PEAK_ARRIVAL_CLOCK (尖峰抵達時間)：顧客抵達時間的平均值（基於常態分佈）。
      * ARRIVAL_DURATION (抵達持續時間)：顧客抵達時間的標準差（基於常態分佈）。
      * RANDOM_SEED (隨機種子)：用於固定或隨機生成顧客抵達時間。

4. 模擬記錄與數據收集：
    - 程式碼使用一個字典 dic_record 來儲存模擬過程中的關鍵數據：  
      * arrival (抵達時間)：顧客的抵達時間。
      * time (事件時間)：發生事件的時間點。
      * queue (排隊人數)：事件發生時的排隊人數。
      * custom_id (顧客ID)：與事件相關的顧客編號。
      * wait_time (等待時間)：顧客開始被收銀服務前的等待時間。
      * done_time (完成時間)：顧客完成排隊和收銀的時間。
      * cashier (收銀員ID)：服務顧客的收銀員編號。

5. 數據分析與視覺化：
    - 模擬結果被轉換為 Pandas DataFrame 進行進一步分析。
    - 利用 plotly 庫生成多種圖表，直觀地呈現模擬結果：
      * 顧客人數分布圖 (Histogram)： 顯示顧客抵達時間的分布。
      * 排隊人數模擬圖 (Scatter Plot / Line Plot)： 呈現隨時間變化的排隊人數、顧客抵達和離開的時間點。
      * 顧客排隊時間甘特圖 (Gantt Chart)： 展示每位顧客從抵達到完成服務的時間段，顏色可能代表等待時間。引用維基百科的介紹：「甘特圖 於1910年由亨利·甘特 (Henry Laurence Gantt) 開發出。顯示專案、進度以及其他與時間相關的系統進展的內在關係隨著時間進展的情況。」
      * 排隊時間分布箱形圖 (Box Plot)： 顯示顧客等待時間的分布統計量（最小值、最大值、中位數、四分位數）。引用維基百科的介紹：「箱形圖 於1977年由美國著名統計學家 約翰·圖基（John Tukey）發明。它能顯示出一組數據的最大值、最小值、中位數、及上下四分位數。」
    - streamlit 介面提供互動式表格 (AgGrid) 供使用者檢視詳細的模擬數據。

## 程式碼結構：
- fn_sim_init(): 初始化模擬環境和參數，生成顧客抵達時間。
- fn_sim_customer(): 模擬顧客的抵達過程。
- fn_sim_cashier(): 模擬收銀員的服務過程。
- fn_sim_main(): 運行模擬主程序。
- fn_sim_fr_st(): 建立 Streamlit 網頁介面，處理使用者輸入和觸發模擬。
- fn_sim_result_render(): 處理模擬結果數據並生成圖表顯示。
- 輔助函數 (e.g., fn_gen_plotly_hist, fn_gen_plotly_scatter, etc.): 用於生成特定類型的 Plotly 圖表。
- fn_2_timestamp(): 將模擬時間（分鐘）轉換為時間戳格式。

## 使用的外部庫：
- streamlit: 用於建立網頁介面。
- streamlit_player: 用於在介面中播放音樂。
- simpy: 用於離散事件模擬。
- datetime: 用於處理時間。
- statistics: 可能用於計算統計數據。
- random: 用於生成隨機數（顧客抵達時間）。
- pprint: 用於美觀地列印數據結構。
- pandas: 用於數據處理和分析。
- st_aggrid: 用於在 Streamlit 介面中顯示互動式表格。
- plotly.graph_objs, plotly.express, plotly.subplots: 用於生成圖表。
