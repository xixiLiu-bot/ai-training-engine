# 让 AI 做出好看 UI 的提示词手册

> 适用：Claude / ChatGPT / Gemini / v0 / Lovable / Bolt / Cursor / Claude Code / Codex / Windsurf
> 用法：先读第 1–3 节建立判断力，之后直接跳到第 6–8 节复制。
> 一句话结论：**好看 = 约束到 token 级 × 一个具体风格锚点 × 一次只做一块屏。**

---

## 目录

1. [为什么 AI 做的界面"一眼 AI 味"](#1-为什么-ai-做的界面一眼-ai-味)
2. [三层约束法（核心方法论）](#2-三层约束法核心方法论)
3. [通用主提示词（复制即用）](#3-通用主提示词复制即用)
4. [十二段式设计 Brief 模板](#4-十二段式设计-brief-模板最有效的单条提示词)
5. [风格锚点关键词库](#5-风格锚点关键词库中英对照)
6. [场景模板 ×12](#6-场景模板12-个可直接改)
7. [微效果提示词 ×12](#7-微效果提示词四段式)
8. [迭代·批评·还原 提示词 ×8](#8-迭代批评还原-提示词)
9. [DESIGN.md 模板（项目级长期约束）](#9-designmd-模板)
10. [工具适配速查](#10-工具适配速查)
11. [参数速查表](#11-参数速查表)
12. [参考资源](#12-参考资源)

---

## 1. 为什么 AI 做的界面"一眼 AI 味"

AI 不是在"设计"，它是在做填空题。你没给约束时，它会填训练数据里出现频率最高的样式——那正是最廉价的那一套。

### 1.1 AI 的十个默认反射（看到就改）

| # | AI 的默认动作 | 为什么廉价 | 换成什么 |
|---|---|---|---|
| 1 | `from-purple-500 to-blue-500` 渐变 | 2021 年模板网标配 | 单色强调 + 中性底；渐变只用于一个装饰元素 |
| 2 | 全站同一个 `rounded-xl` + `shadow-lg` | 层级消失，卡片糊成一片 | 圆角分 3 档（6/12/24），阴影按 elevation 分级且带色相 |
| 3 | 三栏等宽卡片 + 居中图标 + 两行描述 | 功能列表的万能模具 | Bento（1 大 + 2 小）、左右交替、横向滚动 |
| 4 | 居中大标题 + 灰副标题 + 主按钮/次按钮 | 每个落地页都一样 | 左对齐 + 非对称栅格，或标题压在图片上 |
| 5 | 全大写小字 eyebrow：「FEATURES」 | 假装高级的装饰 | 只有内容真是序列时才编号；否则删掉 |
| 6 | 每个 section 都 `fade-in-up` | 滚动时页面在抽搐 | 全页只保留 1 个入场编排 |
| 7 | 纯黑 `#000` + 纯灰 `#999` | 对比生硬、发灰 | 近黑 `#161618`，文本三档 `#F4F4F5/#9B9BA4/#62626A` |
| 8 | 默认 Inter，字号只有 16/24/48 | 无字重、字距、行高设计 | 6 档字号阶梯 + 1 处字重对比（400/600 或 300/700） |
| 9 | 「Supercharge your workflow」式空话 | 文案也是模板的一部分 | 写具体数字、具体动作、具体对象 |
| 10 | p-4 / p-6 / p-8 随机混用 | 视觉节奏塌陷 | 4px 基准网格，间距只从固定档位取 |

### 1.2 四类被写进官方规范的"AI 味"（Anthropic frontend-design 明确点名）

1. **暖奶油底 + 高对比衬线 + 陶土橘**（≈`#F4F1EA` + `#D97757`）——已经被用烂，看到就等于看到 Claude 的默认审美。
2. **近黑底 + 单一荧光酸绿/朱红强调**——每个"科技感"需求都会落到这里。
3. **报纸式排版**：发丝线分隔、零圆角、密排多栏——看着很"设计"，但和内容无关。
4. **SaaS 卡片套装**：所有东西都切成同款圆角卡片、同款 `rgba(0,0,0,.1)` 软阴影、再铺一层渐变光斑。

> 判断标准不是"这四类丑"，而是：**它是被选出来的，还是被默认出来的？**
> 如果同一个提示词换一个产品名，AI 还给你一模一样的页面，那就是默认。

### 1.3 提交前的 12 条自检

- [ ] 页面里所有颜色都能在 token 表里找到出处
- [ ] 字号不超过 6 档，字重不超过 3 档
- [ ] 圆角不超过 3 档，且与组件尺寸成正比
- [ ] 阴影只在需要表达层级的地方出现（不是每张卡都有）
- [ ] 有一处"记忆点"（一个大胆的元素），其余全部安静
- [ ] 全页只有 1 个自动播放的动效
- [ ] 文案里没有任何"赋能/一站式/极致体验"
- [ ] 移动端 390px 宽无横向滚动
- [ ] 键盘 Tab 走一遍，焦点环可见
- [ ] 正文对比度 ≥ 4.5:1
- [ ] 鼠标 hover 态和 active 态都存在（不是只有 hover 变色）
- [ ] 删掉任意一个装饰元素，信息是否有损失？没有就删

---

## 2. 三层约束法（核心方法论）

```
第 1 层  系统级美学底线   →  一次性写进 CLAUDE.md / AGENTS.md，长期生效
第 2 层  风格锚点         →  每次会话指定一个流派（第 5 节）
第 3 层  设计 Token       →  精确到 hex / px，或直接喂 DESIGN.md、截图、branding.json
```

### 第 1 层：系统级美学底线（粘一次，管一个项目）

```text
你是一位有强烈审美主张的产品设计师 + 前端工程师。生成任何界面时，遵守以下底线：

【排版】字号只用 12/14/16/20/24/32/48/64 八档；正文行高 1.6，标题 1.1–1.2；
        标题字距 -0.02em，全大写小字字距 +0.06em；正文行长 ≤ 72 字符。
【色彩】中性底 + 最多 1 个强调色 + 1 个语义色组；不用纯黑纯白（用 #161618 / #FAFAF9）；
        同一页面不超过 3 个色相。
【间距】4px 基准；组件内 8–16px，组件间 24–32px，区块间 64–128px。
【圆角】3 档：小 6–8px（输入/按钮）、中 12–16px（卡片）、大 24–32px（容器/弹层）。
【阴影】按 elevation 三级，全部带轻微色相（不用纯灰 rgba(0,0,0,.1)）。
【动效】只用于①引导注意力 ②表达状态变化；时长 120–320ms；
        缓动 out: cubic-bezier(.22,1,.36,1)，弹性用 cubic-bezier(.34,1.56,.64,1)；
        必须支持 prefers-reduced-motion。
【结构】每屏只有一个视觉重心；装饰元素必须承载信息，否则删掉。

每次交付前，先输出一份简短的 token 方案（颜色 4–6 个 hex、字体 2 个、圆角 3 档、
间距档位、1 个记忆点），再写代码。
```

### 第 2 层：风格锚点（别说"好看点"）

差：`做一个高级的落地页`
好：`Linear 风格：近黑背景、发丝边框、单一靛蓝强调、信息密度高、无阴影靠边框分层`

三种锚定方式，从弱到强：

1. **流派名 + 5 个英文关键词**（够用）
2. **截图**：`按这张图的风格做，只学它的排版逻辑和色彩关系，不要抄内容`
3. **反向提取 token**：用 Firecrawl 的 branding 模式或让多模态模型读截图，输出 JSON：

```text
读这张截图，只输出设计 token，不要解释：
{"colors": {bg, surface, border, text_primary, text_secondary, accent},
 "type": {display_font, body_font, scale: {h1,h2,h3,body,small}, weights:[]},
 "radius": {button, card, input, pill},
 "spacing": {base, section_y, container},
 "shadow": {...}, "notes": "3 条最影响观感的特征"}
```

拿到 JSON 之后：

```text
参考上面的 token JSON，严格遵守配色与间距，重做我的整个项目 UI。
不要引入 JSON 之外的颜色。字体若不可用，替换为最接近的 Google Fonts 并说明。
```

### 第 3 层：设计 Token 契约（决定"像不像设计过"）

在提示词里写死这些，比写十句"要高级"都有效：

```text
背景 #FAFAF9 / 卡片 #FFFFFF / 边框 rgba(23,23,26,.08) / 正文 #17171A / 次要 #6B6B76
强调 #4F46E5（仅用于主按钮、链接、焦点环、单一数据强调）
标题 "Instrument Serif" 600 / 正文 "Inter" 400 / 数字 "JetBrains Mono" 500 tabular-nums
圆角：按钮 8 卡片 16 弹层 24 胶囊 999
阴影：sm 0 1px 2px rgba(23,23,26,.06)；md 0 8px 24px -8px rgba(79,70,229,.18)；
      lg 0 24px 48px -16px rgba(23,23,26,.18)
```

---

## 3. 通用主提示词（复制即用）

### 3.1 中文版（日常首选）

```text
你是资深产品设计师兼前端工程师，为挑剔的客户做界面——他已经否掉了三版"看起来像模板"的方案。

【先想再写】
1. 用一句话说出这个页面的对象、他们此刻要完成的事、这个页面唯一的目标。
2. 给一份极简 token 方案：4–6 个 hex、2 个字体、3 档圆角、间距基准、1 个记忆点。
3. 用 ASCII 画一版线框（不要超过 12 行），确认信息层级。
4. 自问：这套方案换成一个不相关的产品还成立吗？成立就重做，并说明改了什么。

【硬性约束】
- 颜色只能来自 token 方案；禁止蓝紫渐变、禁止霓虹、禁止纯黑纯白。
- 字体只用 2 个家族；字号 ≤ 6 档；字重只用 400/600（需要时加 300 或 700）。
- 间距走 4px 网格；圆角 ≤ 3 档；同一页面阴影 ≤ 3 种。
- 装饰必须承载信息：编号只用于真序列，分隔线只用于真分组。
- 全页只保留 1 处自动动效，其余交互动效 120–320ms。
- 文案写具体：数字、对象、动作。禁止"赋能/一站式/极致/无缝"。
- 移动端 390px 起可用；焦点环可见；正文对比度 ≥ 4.5:1。

【交付】
单个自包含 HTML 文件（内联 CSS + 少量原生 JS），不引外链图片，图形用 CSS/SVG 画。
写完后附 5 行自评：最满意的一处、最没把握的一处、删掉了哪些装饰、以及 390px 下的处理。
```

### 3.2 English 版（对 v0 / Lovable / Cursor 效果更稳）

```text
Act as a senior product designer and frontend engineer. Design and build <PAGE> for <PRODUCT>.

Constraints:
- Palette: exactly 5 colors — bg, surface, border, text, accent. No purple-blue gradients, no neon.
- Type: 2 families max, 6 sizes max, weights 400/600 only. Body line-height 1.6, tracking -0.02em on display.
- Spacing: 4px base grid only. Radius: 3 steps. Shadows: 3 elevations, all tinted.
- One focal point per viewport. Every decorative element must carry information.
- Exactly one non-user-triggered animation. All others 120–320ms, ease-out cubic-bezier(.22,1,.36,1).
- Copy must be specific: real nouns, real numbers. No "supercharge", no lorem ipsum.

Process: state the design intent in one line → list tokens → ASCII wireframe → flag any choice that
would read the same for an unrelated product → then write the code.

Output: one self-contained index.html, no external images (CSS/SVG only), responsive from 390px,
visible focus rings, WCAG AA contrast.
```

### 3.3 项目级固化（省掉每次粘贴）

把 3.1 的「硬性约束」段落到以下任一文件，之后只需说"做一个 XX 页"：

| 工具 | 文件 | 说明 |
|---|---|---|
| Claude Code | `CLAUDE.md` 或 `.claude/skills/` | 支持 skill 自动加载 |
| Codex / Cursor / Windsurf | `AGENTS.md` | 通用事实标准 |
| Cursor | `.cursor/rules/ui.mdc` | 可设为 alwaysApply |
| 任意工具 | `DESIGN.md` | 见第 9 节，最通用 |

---

## 4. 十二段式设计 Brief 模板（最有效的单条提示词）

这是目前公开资料里最有效的单条结构：把"设计"翻译成 AI 能执行的 12 类参数。逐段填，一次只填一个页面。

```text
# 1. 设计意图（Design Intent）
一句话说清气质与矛盾：示例「像便利贴被认真整理过：温暖，但秩序极强」。

# 2. 设计 Token
:root{ 4–6 个颜色 + 2 个字体 + 字号阶梯 + 间距档 + 圆角档 + 阴影 3 档 + 缓动 2 条 + 时长 3 档 }

# 3. 排版（Typography）
表格：角色 | 字号 | 行高 | 字距 | 字重 | 备注（Display/H2/H3/Body-L/Body/Small/Eyebrow）

# 4. 色彩与材质
比例（如 55% 底 / 25% 卡面 / 15% 淡色块 / 5% 强调）+ 每个材质的构成公式（阴影/描边/高光）

# 5. 布局与栅格
容器宽度、栏数与间距、区块垂直节奏、对齐方式、非对称规则

# 6. 页面蓝图（逐段）
导航：高度/结构/交互
Hero：左什么右什么、标题几句、CTA 几个、社会证明
…每一段 3–6 行，写清元素与状态

# 7. 组件规格
表格：组件 | 尺寸/内距/圆角 | 全部状态（default/hover/active/focus/disabled/loading/error）

# 8. 动效与交互
入场编排（谁先谁后、多少 ms、位移多少 px）、环境动效、降级规则

# 9. 响应式
≤980 / ≤640 两档断点分别怎么变（具体到栏数、隐藏谁、缩放多少）

# 10. 可访问性
对比度数值、aria 表达、焦点环、必填提示、tab 顺序

# 11. Do / Don't
各 5–6 条，Don't 要具体到数值（如「圆角不得小于 14px」）

# 12. 交付要求
文件数、是否允许外链资源、自检项、验收宽度（390/768/1440）
```

**填好的迷你样例（SaaS 落地页，可直接用）**

```text
# 1 意图
给财务团队的对账工具：冷、精确、可信。像精密仪器面板，不像创业公司官网。

# 2 Token
--bg #0E1013  --surface #15181D  --border #232830  --text #E8EAED  --muted #8A919E
--accent #2DD4BF（只用于主按钮、关键数字、焦点环）
display "instrument serif" / body "inter" / 数字 "jetbrains mono" tabular-nums
--r-sm 8 --r-md 14 --r-lg 22  --space base 4（8/12/16/24/32/48/72/112）
--ease-out cubic-bezier(.22,1,.36,1) --dur 160/240/420

# 3 排版
Display 64/1.05/-0.03/600 · H2 36/1.15/-0.02/600 · H3 22/1.3/-0.01/600
Body-L 18/1.6/0/400 · Body 15/1.65/0/400 · Small 13/1.5/0/400 · Num 48/1/0/500 mono

# 4 材质
比例 60% #0E1013 / 25% #15181D / 10% 发丝线 #232830 / 5% #2DD4BF
卡片 = surface + 1px border + 顶部 1px 白色 6% 高光；不用外阴影，靠边框分层。

# 5 布局
容器 min(1200px, 100% - 48px)；12 栏 gutter 24；区块 padding-block 112px；
Hero 用 7/5 非对称栅格，文字略宽于视觉元素。

# 6 蓝图
导航 64px：左 logo（16px mono 大写）+ 中 4 个链接（15px muted，hover 变 text）+ 右一个主按钮。
Hero：左 7 栏，H1 两行（第二行以强调色下划线收尾）；副文案 1 句 18px muted；CTA 两个；
      右 5 栏，用纯 CSS 画一个「对账结果」小窗，含 3 行数据 + 一个绿色对勾动画。
证明条：一行 5 个客户 logo，用 SVG 文字占位，灰度 40%，hover 变亮。
能力区：Bento——左一大卡（含小图）+ 右上宽卡 + 右下两张方卡。
特性 ×2：左右交替，图 5 栏 / 文 7 栏。
指标条：3 个大数字（mono，48px，强调色），下挂 13px 说明。
定价：3 档，中间档加深底 + accent 边框 1px + 角标。
FAQ：单列 720px 宽，原生 details，箭头旋转 180°。
尾部 CTA：整块 accent 12% 透明底 + 内发光，只有一个按钮。
Footer：2 列，13px muted。

# 7 组件
主按钮 44px / px 20 / r 8 / accent 底 / 深色字 600；
  hover 亮度 +8%，active translateY(1px)，focus 2px accent 外环 offset 2；
次按钮 44px / 透明 / 1px border / text 色；
输入 44px / r 8 / surface 底 / focus 时边框变 accent + 4px accent 12% 外光晕；
表格行 44px / hover #1A1E24 / 选中左侧 2px accent 竖条。

# 8 动效
唯一编排：Hero 的 eyebrow→H1→副文案→CTA 依次上移 12px 淡入，间隔 70ms，共 420ms。
对勾：进入视口时 stroke-dashoffset 从 100% 到 0，500ms。
其余 hover 全部 160ms。prefers-reduced-motion 下全部瞬时。

# 9 响应式
≤980：Hero 单列（文字在上，小窗在下且缩放 .92），Bento 变 1+2 堆叠，指标 3 列变 1 行 3 小卡。
≤640：导航折叠为图标按钮 + 抽屉；Display 降到 40px；按钮全宽；隐藏证明条 logo 墙。

# 10 无障碍
正文 #E8EAED on #0E1013 = 14.8:1；muted 8A919E = 6.9:1；accent 按钮黑字 = 9.1:1。
数字加 aria-label 完整值；装饰 SVG aria-hidden；details 用原生。

# 11 Do / Don't
Do：用边框而非阴影分层；数字统一 mono 且 tabular；每屏只一个强调色落点。
Don't：不用渐变；不用圆角大于 22px；不用居中标题；不用全大写副标题；卡片不超过 3 种尺寸。

# 12 交付
单个 index.html；无外链图片；自检 390/768/1440 三档；附 5 行自评。
```

---

## 5. 风格锚点关键词库（中英对照）

AI 对英文风格词的响应明显更准。写成 `流派 + 5 个英文关键词`。

| 风格 | 英文关键词（直接粘） | 适合 | 雷区 |
|---|---|---|---|
| Linear 深色科技 | `dark theme, hairline borders, information-dense, single indigo accent, no shadows, keyboard-first` | 工具型产品、后台、开发者产品 | 容易变成"全黑 + 亮蓝"的廉价感 |
| Vercel 极简黑白 | `monochrome, Geist-style type, high contrast, zero decoration, tight grid` | 技术品牌、文档站 | 内容不够强时会显得空 |
| Apple 留白 | `generous whitespace, frosted glass, system font stack, large product imagery, soft depth` | 消费级产品、硬件 | 大图成本高，缺图就崩 |
| Stripe 渐变 | `mesh gradient, precise grid, editorial typography, subtle color wash` | 金融、B2B SaaS | 渐变一多就回到"AI 蓝紫" |
| 杂志编辑 | `editorial, serif display, asymmetric columns, hairline rules, drop caps` | 内容/品牌站、博客 | 不要滥用编号 01/02/03 |
| 瑞士国际主义 | `swiss grid, Helvetica-like, flush-left ragged-right, one red accent, no radius` | 作品集、文化机构 | 排版容错率极低，差 2px 就露怯 |
| 粗野主义 | `brutalist, thick borders, raw layout, monospace, high contrast, sharp corners` | 创意机构、活动页 | 做成"故意丑"就失去可信度 |
| 新粗野 | `neo-brutalism, hard offset shadow, pastel fill, 2px black border, chunky radius` | 年轻向 App、工具 | 颜色超过 4 个就闹 |
| Bento 便当格 | `bento grid, mixed card sizes, rounded 24px, soft borders, one hero tile` | 产品介绍、功能陈列 | 卡片全是同尺寸就退化成三栏 |
| 玻璃拟态 | `glassmorphism, backdrop blur 16px, translucent white 12%, 1px inner highlight` | 音乐、AI、视觉型产品 | 后面没有内容可透，就是灰色糊块 |
| 轻黏土 | `claymorphism, puffy surfaces, inner top highlight, long tinted shadow, big radius` | 消费金融、儿童/教育 | 容易甜到腻 |
| 新拟物 | `neumorphism, dual shadows, extruded soft surface, low contrast` | 硬件控制面板 | 无障碍灾难，慎用 |
| 极光/网格渐变 | `aurora gradient, mesh blur, grain overlay, deep navy base` | AI 产品首屏 | 90% 的 AI 站都长这样 |
| 复古 Y2K | `Y2K, chrome type, pixel icons, acid pastel, scanlines` | 潮流品牌、活动 | 生命周期极短 |
| 赛博/终端 | `terminal aesthetic, monospace UI, scanline, phosphor green, ASCII dividers` | 开发者工具 | 正文可读性差 |
| 暗金奢华 | `luxury, near-black, gold hairline, wide tracking, serif display, lots of negative space` | 高端服务、地产 | 金色饱和度一高就土 |
| 自然有机 | `organic, warm paper texture, hand-drawn shapes, muted earth tones, generous radius` | 健康、食品、生活方式 | 容易变成"奶油 + 陶土"套路 |
| 动效排版 | `kinetic typography, scroll-driven, masked text reveal, sticky scenes` | 品牌故事页 | 性能与可访问性风险高 |

**组合公式**（比单一风格更耐看）：

```
主风格（结构） + 一个材质（表面） + 一个反差点（记忆点）
例：瑞士栅格（结构）+ 纸张纹理与暖白（材质）+ 一处克莱因蓝大色块（记忆点）
例：Linear 深色（结构）+ 玻璃拟态（材质）+ 一条手绘荧光黄标注线（记忆点）
```

---

## 6. 场景模板（12 个，可直接改）

> 每个模板都遵循：**风格锚点 → Token → 结构 → 该场景特有的硬要求 → 输出与自检**。
> 把 `<>` 里的内容替换掉就能用。

### 6.1 SaaS 落地页

```text
做 <产品名> 的落地页，一个单文件 index.html（内联 CSS，无外链图片，图形用 CSS/SVG）。
风格：Stripe 式精确栅格 + 编辑感排版（mesh gradient 只用在一个装饰块）。

Token：bg #FFFFFF / surface #F7F8FA / border #E6E8EC / text #0B0F17 / muted #5A6473 / accent #4F5BFF
字体：标题 Inter Tight 600 -0.02em，正文 Inter 400 16/1.6，数字 JetBrains Mono tabular
圆角：按钮 10 / 卡片 16 / 容器 24；阴影：只有"截图窗"和"定价推荐档"有

结构：导航 72px → Hero（左文 7 栏 / 右产品窗口 5 栏）→ Logo 墙 → Bento 能力区（1 大 2 小）
      → 特性交替 ×2 → 指标条（3 个大数字）→ 定价 3 档 → FAQ → 尾部 CTA → Footer

硬要求：
- Hero 标题两行，只把最后一行的 1 个词上强调色，禁止整句渐变字
- 产品窗口用 div 画：标题栏 3 个圆点 + 侧栏 4 行 + 主区 2 张卡 + 1 条折线（SVG path）
- 禁止三栏等宽"图标+标题+描述"，改用 Bento
- 全页只有 1 个入场编排（Hero 四元素 70ms 交错）
- 390px 下：单列、按钮全宽、Logo 墙隐藏、标题 40px
交付后自评 5 行：记忆点是什么、删了哪些装饰、390px 怎么处理。
```

### 6.2 数据看板 Dashboard

```text
做一个 <行业> 运营看板，单个 HTML 文件。
风格：Linear 深色 + 高信息密度（信息密度就是高级感的来源之一）。

Token：bg #131417 / panel #1A1C20 / border #26282E / text #EDEEF0 / muted #868B96 / accent #6E8BFF
      positive #4CC38A / negative #F2555A / warn #E0A33E
字体：正文 Inter 13/1.5，标题 15/600，所有数字 JetBrains Mono tabular-nums
圆角：卡片 10 / 标签 6 / 图表 0；阴影：无（靠 1px border 分层）

结构：左侧栏 232px（logo + 6 个导航 + 底部用户）→ 顶栏 56px（面包屑 + 搜索 + 日期范围 + 头像）
      → 4 个 KPI 卡（数字 32px mono + 环比小箭头）→ 主折线图（占 8 栏）+ 右侧环形图（4 栏）
      → 明细表（可排序表头，行高 40px）

硬要求：
- 图表用 SVG 手绘，禁止引第三方库；折线要有网格线（横向 4 条，色 #22252B）
- KPI 卡里只有 1 个数字用 accent 色，其余用 text
- 表格：hover 行底色 #1F2228，选中行左侧 2px accent 竖条，状态列用小圆点 + 文字
- 侧栏当前项：底色 #212429 + 左侧 2px accent
- 不要圆角大卡片堆叠；不要彩色背景区块
- 1440/1024/768 三档：1024 收起侧栏为 64px 图标条，768 顶部变横向滚动 tab
```

### 6.3 移动端 App 界面（iOS 风）

```text
做一个 <App 功能> 的移动端界面，展示在 390×844 的手机框里（HTML + CSS）。
风格：iOS 原生感 —— 大标题、分组列表、细腻的层级、克制的色彩。

Token：bg #F2F2F7 / card #FFFFFF / separator rgba(60,60,67,.12) / text #000 / secondary #8E8E93
      accent #0A84FF / destructive #FF3B30
字体：-apple-system, "SF Pro Text"（回退 Inter）；大标题 34/700/-0.02em；正文 17/1.4；说明 13
圆角：卡片 12 / 按钮 12 / 头像 50%；阴影：仅浮层 0 8px 32px rgba(0,0,0,.12)

结构：状态栏（时间 + 信号电池）→ 大标题 + 右上角头像 → 搜索框 → 分组卡（3 组，每组行高 52px，
      左图标 28px 圆角方块 + 标题 + 右侧 chevron）→ 底部 TabBar 4 项（1 项选中用 accent）

硬要求：
- 分割线用 0.5px，且只在行与行之间，不要外框
- TabBar 用 backdrop-filter: blur(20px) + 半透明白
- 点击态：整行背景变 rgba(0,0,0,.04)，不要缩放
- 一处记忆点：某个卡片用渐变描边或一张插图位
- 所有可点区域 ≥ 44px；正文不用小于 13px
```

### 6.4 电商商品页

```text
做 <品类> 商品详情页。
风格：杂志编辑 × 电商转化（大图 + 呼吸感留白，信息密度低但有力量）。

Token：bg #FBF9F6 / ink #1A1714 / line #E4DED5 / accent #A8442A（砖红，只用于价格与购买）
字体：标题 "Fraunces" 500 衬线（有性格），正文 Inter 400，规格数字 mono
圆角：图片 4（近乎直角，显质感）/ 按钮 2（更利落）/ 标签 999
阴影：无；用 1px line 与留白分层

结构：顶部细导航（含购物袋计数）→ 左侧图片廊（主图 + 4 张缩略图竖排）→ 右侧购买区
      （品名/副标题/价格/评分/颜色 4 选/尺码 5 选/数量/加入购物袋/物流说明/3 条折叠详情）
      → 横向"搭配推荐" → 品牌故事左右分栏 → 评价摘要 → Footer

硬要求：
- 价格 28px 衬线，强调色；划线原价 14px muted
- 颜色选择用 32px 圆形色块，选中加 2px outline offset 3px
- 尺码按钮 44×44 方形，选中反色；缺货加斜线（CSS linear-gradient）
- 加入购物袋按钮 52px 高，hover 时整按钮上移 1px
- 商品图用 CSS 渐变 + 纹理模拟，不要外链
- 移动端：图片廊变横向滑动 + 圆点指示器
```

### 6.5 后台 / 表格密集型

```text
做 <业务> 的列表管理页。
风格：Linear / Notion 混合 —— 极简、行内编辑、键盘优先。

Token：bg #FFFFFF / header #FAFAFB / border #EBECEF / text #1C1E21 / muted #767B85 / accent #5E6AD2
行高 40px，表头 36px（12px 全大写 +0.06em muted）
字体：正文 13，表头 11，ID 与时间 mono

结构：页面标题 + 右侧主操作 → 筛选栏（搜索 280px + 3 个下拉 + 视图切换）
      → 表格（复选框 / 名称 / 状态 / 负责人 / 更新时间 / 操作）
      → 底部：已选 3 条 + 批量操作 + 分页

硬要求：
- 表头 sticky；列宽固定，名称列可伸缩
- 状态用「小圆点 + 文字」，不用彩色胶囊
- 操作列图标按钮，hover 才显形
- 空状态：居中一个细线 SVG 插画 + 一句解释 + 一个主按钮，不要 emoji
- 批量选择后顶部出现浮动操作条（底部居中，圆角 12，阴影 —— 全页唯一阴影）
```

### 6.6 AI 对话 / Agent 界面

```text
做 <AI 产品> 的对话界面。
风格：克制、专注、有"工具感"——不做发光球体，不做粒子背景。

Token：bg #101114 / panel #16181D / border #24272E / text #ECEEF2 / muted #8A909C
      accent #B4A0FF（仅用于光标、发送按钮激活态、引用高亮）
字体：正文 15/1.7（对话要更松），代码 JetBrains Mono 13，标题 13 全大写 +0.08em muted

结构：左 260px 会话列表（今天/昨天分组，当前项左侧 2px accent）→ 主区
      （顶部模型选择 + 主区最大宽 720px 居中）→ 消息流 → 底部输入框（圆角 16，含附件/发送）
      → 右侧 320px 可选上下文面板（可折叠）

硬要求：
- 用户消息右对齐、无气泡（只有底色 #1E2128 + 圆角 14，最大宽 80%）
- 助手消息左对齐、无气泡、无头像，仅文字流；引用块左侧 2px accent
- 消息出现：仅首条做 200ms 淡入；流式输出用光标闪烁 1s steps(2)
- 输入框 focus 时边框 accent + 3px accent 10% 光晕
- 禁止：渐变球、玻璃卡片、彩色头像、聊天气泡尾巴
```

### 6.7 定价页

```text
做 <产品> 定价页。
风格：清晰优先——三档差异必须一眼看出，不靠颜色堆。

Token：bg #FFFFFF / card #FFFFFF / border #E5E7EB / text #111827 / muted #6B7280 / accent #2563EB
字体：档位名 20/600，价格 48/700 mono tabular，说明 14/1.6

结构：标题 + 月付/年付切换（年付显示 -20% 小标）→ 3 档卡（中间档 border 2px accent + 顶部
      "最受欢迎"角标 + 轻微上浮 -12px）→ 功能对比表（首列冻结）→ FAQ → 企业定制条

硬要求：
- 每档只列 5 条功能，不列全量；全量放对比表
- 价格数字用 tabular-nums，货币符号小一号上标
- 年付切换时价格数字做 160ms 数字滚动（不是整体淡入）
- 中间档按钮实心，两侧描边；三档按钮同宽同高
- 对比表：有/无用 ✓ 与 —（不用彩色叉），分组行加浅底
- 移动端：卡片纵向堆叠，推荐档放第一位
```

### 6.8 登录 / 注册

```text
做 <产品> 的登录页。
风格：极简、可信、零干扰。左表单右品牌面板（≥1024px），以下单列。

Token：bg #FAFAF9 / panel #FFFFFF / border #E7E5E4 / text #1C1917 / muted #78716C / accent #1C1917（黑白主色）
字体：标题 28/600 -0.02em，正文 14/1.5，输入 15

结构：左上 logo → 表单（标题 + 副标题 + 邮箱 + 密码 + 忘记密码 + 主按钮 + 分隔线 "或"
      + 2 个 SSO 按钮 + 底部注册链接）→ 右侧品牌面板（一句价值主张 + 一条客户引言 + 3 个 logo）

硬要求：
- 输入框 48px 高，label 常驻在上方（不要 placeholder 当 label）
- 密码框右侧有显示/隐藏眼睛图标
- 错误态：边框 #DC2626 + 下方 13px 红字 + 图标，不用 toast
- 主按钮 loading 态：文字变 3 点脉冲，宽度不变
- 右侧面板用极浅的斜向网格（CSS repeating-linear-gradient）做质感，不要大图
- 表单区最大宽 380px，整体垂直居中偏上（padding-top 12vh）
```

### 6.9 个人作品集

```text
做 <职业> 的作品集首页。
风格：编辑感 + 大胆排版（排版本身就是作品的一部分）。

Token：bg #F5F2EC / ink #141210 / line #D8D2C6 / accent #1B3A2F（深墨绿）
字体：Display "Instrument Serif" 400（超大，clamp(56px, 9vw, 132px)），正文 Inter 400 17/1.6，
      标签 mono 11 大写 +0.1em
圆角：0（全直角）；阴影：无；分割用 1px line

结构：极简导航（名字 + 3 个链接，字号 13）→ 巨型标题（两行，一行衬线一行无衬线形成对比）
      → 一句自我介绍（max-width 520px）→ 项目列表（不用卡片：每行是一个项目，
      左 序号 mono / 中 项目名 28px / 右 年份 + hover 时整行右移 8px 并出现 →）
      → 关于（两栏：左标题右正文）→ 联系方式（大字邮箱，hover 下划线动画）

硬要求：
- 项目行 hover：背景从透明到 rgba(20,18,16,.03)，箭头从 opacity 0→1 位移 0→8px
- 全页只有 1 个页面入场：标题遮罩上滑（clip-path 从 inset(0 0 100% 0) 到 inset(0)）
- 移动端项目行改为两行布局，序号移到项目名上方
- 禁止：卡片网格、头像圆形、emoji、渐变
```

### 6.10 空状态 / 错误 / 加载

```text
为一组页面状态做 3 个组件（同一个 HTML 文件里横向展示）：
① 空状态 ② 加载骨架 ③ 错误页

风格：与产品一致的中性体系，不用插画网站素材，全部 CSS/SVG 手绘。

Token：bg #FFFFFF / surface #F7F7F8 / border #E8E8EA / text #18181B / muted #71717A / accent #4F46E5

① 空状态：居中，一个 120×120 的细线 SVG（2px 描边，muted 色，只画必要 5 条线）+
   标题 18/600 + 说明 14 muted 两行 + 1 个主按钮 + 1 个文字链接。
   文案写"还没有 <具体对象>，<一句怎么创建>"，禁止"暂无数据"。
② 加载：骨架屏，3 行（60%/90%/40% 宽），底色 #EFEFF1，高光用 CSS 动画
   linear-gradient 扫过 1.4s infinite；行圆角 6。
③ 错误页：大号 40px mono 错误码 + 一句人话解释 + 2 个按钮（重试 / 返回）+ 一行小字
   "错误码 XXX-XXX"。

硬要求：三者共用一套间距；错误页不用红色大面积，只用 1 个红点/图标；骨架屏必须有
prefers-reduced-motion 降级为静态。
```

### 6.11 设置页

```text
做 <产品> 设置页。
风格：Notion 式的安静表单——分组清晰、说明克制、无卡片堆叠。

Token：bg #FFFFFF / sidebar #FAFAFA / border #ECECEE / text #1F1F23 / muted #75757F / accent #5E6AD2

结构：左侧 200px 设置导航（个人资料/账户/通知/团队/账单/API，当前项左侧 2px accent）
      → 右侧 max-width 640px 内容 → 每个分组：24px 标题 + 13px 说明 + 一组行
      → 行：左 label 14/500 + 可选 13 muted 说明；右控件（开关/下拉/输入）

硬要求：
- 开关 44×26，开启态 accent，滑块 22px 白色 + 1px 阴影，transition 160ms
- 危险操作单独分组，标题用 #DC2626，按钮为描边红
- 保存不是即时生效的项，底部出现 sticky 保存条（背景白 + 上边框 + 右对齐两按钮）
- 每行分隔用 1px border-bottom，最后一行去掉
```

### 6.12 移动端 H5 活动页

```text
做 <活动> 的 H5 单页（390px 设计稿，桌面端居中显示为手机宽度）。
风格：<潮流/国潮/极简> —— 一屏一个信息，强对比，适合竖滑。

Token：bg <#0B0B0F> / text #FFFFFF / accent <#FF4D2E> / secondary <#FFE566>
字体：大标题用有性格的展示字体（clamp 32–44px，行高 1.05，字距 -0.03em），正文 15/1.7

结构：首屏（活动名 + 一句钩子 + 日期 + 倒计时 + 主 CTA）→ 3 个卖点（每屏一个，大字 + 小注）
      → 规则说明（折叠）→ 报名表单（3 个字段）→ 底部固定 CTA 条（高 64 + 安全区）

硬要求：
- 滚动驱动：卖点文字在进入视口 60% 时从 opacity .2 / translateY(24px) 到正常，
  背景装饰（大圆/粗线/网格）以 0.15 倍速视差移动
- 底部 CTA 固定，滚动超过 1 屏才出现（opacity + translateY 过渡 240ms）
- 首屏倒计时用 mono tabular，数字切换只做位移动画不做缩放
- 表单输入 52px 高，大圆角 14，focus 时 accent 边框
- 禁止：整屏渐变紫、模糊光斑、emoji 图标堆叠
```

---

## 7. 微效果提示词（四段式）

**结构：效果名称 + 触发方式 + 技术实现 + 具体参数。** 只说"加点动效"必然得到一堆 `fade-in`。

### 7.1 卡片光斑跟随

```text
效果：鼠标在卡片上移动时，卡片内部有一团跟随鼠标的柔光。
触发：mousemove（节流到 rAF）。
实现：卡片 position:relative + overflow:hidden；伪元素 ::before 用
      background: radial-gradient(240px circle at var(--x) var(--y), rgba(255,255,255,.14), transparent 60%)；
      JS 用 getBoundingClientRect 更新 --x/--y。
参数：光斑半径 240px，透明度 14%，无过渡（跟手）；移出时 opacity 200ms 归零。
```

### 7.2 渐变文字（克制版）

```text
效果：只有 H1 的最后 1–2 个词是渐变字。
实现：display:inline-block；background:linear-gradient(120deg,#111 20%,#4F46E5 60%,#111 100%)；
      -webkit-background-clip:text；color:transparent。
参数：渐变跨 3 个色标，首尾同色，避免整体发灰；不要给整段文字上渐变。
```

### 7.3 毛玻璃导航（滚动渐显）

```text
效果：导航在滚动 64px 后从透明变成毛玻璃。
实现：backdrop-filter: blur(14px) saturate(140%)；背景 rgba(255,255,255,.72)；
      顶部 1px 内阴影 rgba(255,255,255,.6) 作为高光边。
参数：opacity 0→1，translateY -8px→0，时长 220ms，ease-out；初始态不得拦截点击。
```

### 7.4 数字滚动

```text
效果：滚动到指标区时，数字从 0 递增到目标值。
实现：IntersectionObserver 触发一次；requestAnimationFrame + easeOutExpo 插值；
      用 tabular-nums 防止宽度跳动。
参数：时长 1100ms；保留原有小数位；千分位用 toLocaleString；减少动效时直接显示终值。
```

### 7.5 磁吸按钮

```text
效果：鼠标靠近按钮时，按钮整体轻微被"吸"向光标。
实现：mousemove 计算光标相对按钮中心的偏移，translate 限制在 ±6px，内部文字反向位移 40%。
参数：跟随系数 0.25；离开时 300ms 弹性回位 cubic-bezier(.34,1.56,.64,1)；
      触屏设备与 prefers-reduced-motion 下禁用。
```

### 7.6 文字逐行遮罩上滑

```text
效果：标题按行从下方"被揭开"。
实现：每行包一层 overflow:hidden 的 span，内层 transform:translateY(110%)；
      进入视口后 translateY(0)。
参数：每行 600ms，间隔 80ms，ease cubic-bezier(.22,1,.36,1)；
      不用透明淡入，遮罩位移更干净。
```

### 7.7 图片视差揭示

```text
效果：滚动时图片比容器慢速移动，形成纵深。
实现：容器 overflow:hidden；内部 img 高度 120%，top:-10%；
      JS 按容器在视口中的比例设置 translateY(±10%)。
参数：位移幅度 10%，无缓动（跟随滚动）；requestAnimationFrame 内读取，避免 layout thrash。
```

### 7.8 堆叠卡片滚动

```text
效果：多张卡片滚动时依次吸附堆叠，上一张缩小淡出。
实现：每张卡 position:sticky; top: 24px + index*8px；滚动时用 scroll 进度控制 scale 与 opacity。
参数：scale 1→0.94；opacity 1→0.6；卡片间距 16px；容器高度 = 卡片数 × 80vh。
```

### 7.9 极光背景（别做成 AI 站）

```text
效果：首屏背景有缓慢流动的柔和光斑。
实现：3 个 position:absolute 的圆，background: radial-gradient，filter: blur(80px)，
      用 CSS @keyframes 改变 translate 与 scale。叠加一层 3% 噪声（SVG feTurbulence data-uri）。
参数：单个光斑半径 40vw；循环 18–28s，各自不同 delay；整体 opacity ≤ 0.5；
      主色只取 2 个色相，第三个用同色系更深值。
```

### 7.10 自定义光标（谨慎）

```text
效果：桌面端用一个 10px 圆点替代默认光标，hover 可点元素时放大到 28px 并显示文字。
实现：一个 fixed 的 div + mix-blend-mode: difference；JS 用 lerp 跟随（0.18）。
参数：直径 10px，跟随延迟 0.18；可点元素放大到 28px + 背景反色；
      触屏、减少动效、以及表单输入区必须恢复系统光标。
```

### 7.11 骨架屏

```text
效果：内容加载时显示与真实布局同尺寸的灰块，带扫光。
实现：块状 div 用 #EFEFF1，伪元素 linear-gradient(90deg, transparent, rgba(255,255,255,.7), transparent)
      横向移动；容器 aria-busy="true"。
参数：扫光 1.4s infinite linear；圆角与真实组件一致（差 2px 就会"跳"）。
```

### 7.12 滚动进度指示

```text
效果：页面顶部一条 2px 的阅读进度线。
实现：position:fixed; top:0; height:2px; width:calc(var(--p)*100%)；scroll 事件更新 --p。
参数：accent 色；顶部带一点外发光；到 100% 时不消失（避免跳动）；不遮挡内容点击。
```

---

## 8. 迭代·批评·还原 提示词

好看与否，80% 取决于第二轮怎么骂。下面 8 条按使用频率排序。

### 8.1 自我批评（每轮必加）

```text
现在以设计总监身份审这份产出。列出最"廉价"的 5 处，按严重程度排序，每处写：
① 具体位置 ② 为什么廉价（说出它对应哪个常见的 AI 默认套路）③ 具体改法（给出数值）。
不要改代码，先给清单。
```

### 8.2 去 AI 味重写

```text
这份设计有 AI 默认痕迹。逐条检查并修掉：
- 蓝紫渐变 / 霓虹色 / 纯黑纯白
- 同款圆角 + 同款阴影铺满全页
- 三栏等宽卡片、居中标题、全大写小标题
- 每个 section 都有 fade-in-up
- 空话文案
改完后用一句话说明：现在的记忆点是什么？其余部分为它让了什么路？
```

### 8.3 还原参考站风格（不抄内容）

```text
这是参考站截图。只提取它的：栅格比例、字号层级关系、间距节奏、色彩关系、圆角与阴影逻辑。
用这套逻辑重做我的页面内容。明确禁止：复制它的文案、图标、插画、品牌色值。
输出前先用 6 行说明你提取到的规则，再写代码。
```

### 8.4 三方向对比

```text
给我 3 个视觉方向，共享同一套信息结构，但在这三点上必须不同：
① 排版重心（左对齐/居中/非对称）② 材质（纯色/线框/玻璃）③ 强调色强度（无/一点/大色块）。
每个方向只写 12 行规格 + 一句适用场景，不要写代码。我选一个你再实现。
```

### 8.5 像素级走查

```text
按这份清单走查并逐项修复，每项给出改前→改后的数值：
1 标题与正文的字号比是否 ≤ 3:1
2 相邻区块间距是否一致（±4px 内）
3 所有卡片圆角是否同源
4 按钮高度是否统一（同行内）
5 图标视觉重量是否一致（描边宽度统一）
6 灰色文字是否都来自同一个 muted token
7 hover 态是否都有，且不是只有颜色变化
8 焦点环是否在深/浅底上都可见
```

### 8.6 响应式抢救

```text
现在把它压缩到 390px：
- 不允许横向滚动
- 文字不允许小于 13px
- 可点区域 ≥ 44px
- 表格改卡片列表；多栏改单列；横向导航改抽屉
- 标题用 clamp() 平滑缩放，不要跳变
逐条说明你改了什么，以及为什么这样改而不是隐藏内容。
```

### 8.7 可访问性审计

```text
审计并修复：
- 所有正文对比度 ≥ 4.5:1，大字与 UI 元件 ≥ 3:1（给出每处的实测比值）
- 所有图标按钮有 aria-label
- 装饰元素 aria-hidden
- 动态内容（数字、提示、加载）用 aria-live="polite"
- Tab 顺序符合视觉顺序，焦点环 2px 且 offset ≥ 2
- 所有动效支持 prefers-reduced-motion
```

### 8.8 换皮测试（判断"是否只是模板"）

```text
保持结构不变，把这个设计换成一个完全不同的产品（<另一个行业>）。
如果你只需要改文案就成立，说明结构太通用——请指出哪 3 处结构是该产品独有的，
并把不够独特的地方改成只有这个产品才会有的版式。
```

---

## 9. DESIGN.md 模板

落一个文件在项目根目录，之后每个会话的第一句话就是「遵循 DESIGN.md」。这是跨工具最通用的做法（Cursor / Claude Code / Codex / v0 / Lovable 都吃）。

```markdown
---
name: <产品名> 设计系统
style: <一句话风格，例：dark, compact, keyboard-first>
colors:
  primary: "#5E6AD2"
  background: "#161618"
  surface: "#1F1F21"
  surface-raised: "#2A2A2C"
  border: "#2F2F31"
  text-primary: "#F4F4F5"
  text-secondary: "#9B9BA4"
  text-muted: "#62626A"
  success: "#4CAF76"
  error: "#E5534B"
typography:
  display: "<字体>, system-ui"
  body: "<字体>, system-ui"
  mono: "JetBrains Mono, monospace"
  scale: { h1: "1.5rem/600", h2: "1.125rem/600", body: "0.875rem/400", small: "0.8125rem/400" }
spacing: { base: 4px, steps: [2,4,8,12,16,24,32,48,64] }
radius: { button: 6px, card: 8px, input: 6px, tag: 4px }
---

# 视觉方向
2–3 句。说清密度、气质、唯一强调色的用法。

# 布局
侧栏宽度 / 内容最大宽 / 行高 / 区块间距。给数值。

# 组件
按钮：四种（primary/secondary/ghost/danger），给出 Tailwind class 或 CSS。
卡片：底色 + 边框 + 圆角 + 内距。
列表行：高度 + hover + active。
输入：高度 + 边框 + focus 表现。
状态：用「彩色圆点 + 文字」，不用胶囊。

# 间距规则
图标与文字 4px；列表项之间 8px；面板内分区 16px；主分区 24px；末元素不留 margin-bottom。

# Do
- 默认深色，`#161618` 是唯一底色
- 用边框表达层级，不用阴影
- 数字与 ID 用 mono
- 密度优先：拿不准时把 padding 减 4px

# Don't
- 不用白底
- 不用 box-shadow 做 elevation
- 按钮不用 rounded-full（保持 6px）
- 不引入 token 之外的颜色
- 不做装饰性元素

# Agent 指令
1. 所有背景必须来自上面的深色调色板
2. 边框用 #2F2F31，不要用 Tailwind 的 border-gray-*
3. 工具栏优先图标按钮，而非带字按钮
4. 状态一律「色点 + 文字」
5. 列表行 hover: bg-[#1F1F21]，active: bg-[#2A2A2C]
```

**怎么快速得到一个项目的 DESIGN.md**：让多模态模型读你的截图或现网站，按上面字段输出；或用 Firecrawl 的 branding 模式抓目标站，再让 AI 转成这个结构。

---

## 10. 工具适配速查

| 工具 | 提示词偏好 | 关键技巧 |
|---|---|---|
| **v0** | 短、结构化、点名组件 | 先发一段"设计系统提示"（色板 hex + 圆角 + 字号层级）并保存；之后每次生成都在同一系统内 |
| **Lovable / Bolt** | 偏向"产品需求"叙述 | 第一轮只要结构，第二轮再贴 token；一次只让它改一个区块 |
| **Claude Code / Codex / Cursor** | 吃长上下文与文件约束 | 把第 3 节的底线写进 `CLAUDE.md` / `AGENTS.md`；给它 `DESIGN.md` 比给它形容词有用 |
| **ChatGPT / Gemini 直接出 HTML** | 需要强自检指令 | 末尾必加"输出前自评 5 行"和"单文件、无外链图片" |
| **shadcn/ui + Tailwind** | 最稳的落地组合 | 让 AI 只改 token（CSS 变量）和布局，别让它重写组件内部；组件源码在项目里，可改 |
| **Tailwind v4** | 用 `@theme` 定义 token | 让 AI 先写 `@theme` 块，再写页面，一致性最好 |
| **Magic UI / Motion** | 补动效 | 直接点名组件（如 ShimmerButton），比让它从零写动画稳 |
| **Figma MCP / 截图** | 视觉锚定最强 | 截图 + "只学排版与色彩关系，不抄内容" |

> 通用原则：**组件库只选一套。** shadcn 和 Ant Design 混用会让 AI 在两套设计语言间摇摆，输出更容易乱。

---

## 11. 参数速查表

### 字号阶梯（任选一套，别自创）

```
小而密：12 / 13 / 14 / 16 / 20 / 24 / 32 / 48
常规：  14 / 16 / 18 / 20 / 24 / 32 / 48 / 64
展示型：16 / 18 / 20 / 24 / 36 / 56 / 80 / 112
```

- 正文行高 1.5–1.7；标题 1.05–1.25
- 标题字距 `-0.02em` 到 `-0.035em`（越大负得越多）
- 全大写小字字距 `+0.06em` 到 `+0.1em`
- 正文行长 45–75 字符，常用 65

### 间距

- 基准 4px；档位只用 4/8/12/16/24/32/48/64/96/128
- 组件内 8–16；组件间 24–32；区块间 64–128
- 密集产品（后台）：区块间 32–48

### 圆角（三档制）

| 档 | 值 | 用于 |
|---|---|---|
| 小 | 6–8px | 按钮、输入、标签 |
| 中 | 12–16px | 卡片、面板、下拉 |
| 大 | 24–32px | 弹层、大容器、图片 |
| 胶囊 | 999px | 只用于真实胶囊场景（chip、头像、开关） |

> 圆角应与尺寸成正比：24px 卡片配 8px 按钮，不要全站一个值。

### 阴影（三级 + 带色相）

```css
--shadow-sm: 0 1px 2px rgba(23,23,26,.06);
--shadow-md: 0 4px 12px -2px rgba(23,23,26,.08), 0 2px 4px -2px rgba(23,23,26,.06);
--shadow-lg: 0 24px 48px -16px rgba(23,23,26,.18), 0 8px 16px -8px rgba(23,23,26,.10);
/* 带强调色色相（更贵） */
--shadow-accent: 0 12px 32px -12px rgba(79,70,229,.45);
```

深色界面的"阴影"应改为：`inset 0 1px 0 rgba(255,255,255,.06)` 顶部高光 + 更深的底色。

### 动效

| 用途 | 时长 | 缓动 |
|---|---|---|
| hover / 颜色变化 | 120–160ms | `ease-out` |
| 展开 / 收起 | 200–280ms | `cubic-bezier(.4,0,.2,1)` |
| 入场编排 | 400–700ms | `cubic-bezier(.22,1,.36,1)` |
| 弹性反馈 | 300–500ms | `cubic-bezier(.34,1.56,.64,1)` |
| 环境循环 | 6–28s | `ease-in-out` |

- 位移量：入场 8–24px；hover 抬升 1–4px（超过 6px 就夸张）
- 同屏自动动效 ≤ 1 个；单页高级效果 ≤ 2–3 个

### 对比度

- 正文 ≥ 4.5:1；大字（≥24px 或 ≥19px 粗体）与 UI 边界 ≥ 3:1
- 深色底上，`#FFFFFF` 文字配 `#0E1013` 底 ≈ 17:1，很安全；`#8A919E` 灰字 ≈ 6.9:1
- 浅色底上，`#6B7280` 配 `#FFFFFF` ≈ 4.8:1，刚好过线；再浅就不合格

---

## 12. 参考资源

**方法论 / 规范**

- Anthropic 官方 `frontend-design` skill（去 AI 味的第一手依据）：[claude-plugins-official](https://github.com/anthropics/claude-plugins-official/blob/main/plugins/frontend-design/skills/frontend-design/SKILL.md)
- 三层约束法（系统提示词 → 风格锚点 → 品牌 DNA）：[为什么你用 AI 做的界面一眼就看出 AI 味](https://news.qiniu.com/archives/1784773065240)
- 做 UI 的提示词模板 + 组件库选型：[不会设计也能搭出高级感网页](https://www.uisdc.com/design-with-ai-3)

**可直接抄的成品提示词库**

- UI Atlas / ui-design-prompts（12 章式完整设计规格，含 demo）：[qianye60/ui-design-prompts](https://github.com/qianye60/ui-design-prompts)
- Superdesign（140+ 条按真实使用量排序的 UI 提示词，CC0）：[superdesigndev/superdesign-prompts](https://github.com/superdesigndev/superdesign-prompts)
- DESIGN.md 目录（Linear / Vercel / Stripe / shadcn 等设计系统文件）：[dimabraven/design-md](https://github.com/dimabraven/design-md)
- 中文 UI 风格提示模板库：[TonnyWong1052/UI-Prompt](https://github.com/TonnyWong1052/UI-Prompt)

**组件与素材**

- shadcn/ui：https://ui.shadcn.com
- Magic UI 动效组件：https://magicui.design
- Tailwind CSS：https://tailwindcss.com

---

## 附：一页速用卡

```text
【三条铁律】
1. 颜色、字号、圆角、间距，必须给具体值——形容词不产生设计。
2. 一个页面一个记忆点，其余全部安静。
3. 一次只做一块屏；整页生成 = 牺牲视觉细节。

【粘贴顺序】
第 1 句：角色 + 客户已经否掉三版模板方案
第 2 句：风格锚点（流派 + 5 个英文关键词，或截图）
第 3 段：Token（4–6 hex / 2 字体 / 3 圆角 / 4px 网格）
第 4 段：结构（逐段，含每段元素与状态）
第 5 段：禁止清单（蓝紫渐变、等宽三栏、居中标题、全页 fade-in、空话文案）
第 6 段：先出 token + ASCII 线框，我确认后再写代码
第 7 段：写完自评 5 行

【每轮追加】
"以设计总监身份挑出最廉价的 5 处，给位置、原因、具体改法。"
```
