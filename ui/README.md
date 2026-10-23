# AI Recogniser — UI

Прототип интерфейса M0: React 18.3 + TypeScript + Fluent UI v9 + Vite.

Интерфейс полностью русскоязычный, работает офлайн: только системные шрифты и CSS-градиенты,
никаких внешних шрифтов, картинок и сетевых запросов (кроме локального ядра).

## Команды

```powershell
npm.cmd install
npm.cmd run dev      # Vite на http://127.0.0.1:5173, /api и /healthz проксируются на :8765
npm.cmd run build    # tsc --noEmit + production-сборка в ui/dist
npm.cmd run preview  # локальный просмотр production-сборки
```

На Windows используйте `npm.cmd`: PowerShell ExecutionPolicy может блокировать `npm.ps1`.

Сборка попадает в `ui/dist` — именно её отдаёт ядро (`AppPaths.ui`). Перед запуском приложения
целиком выполните `npm.cmd run build`.

## Подключение к ядру

- По умолчанию запросы идут на тот же origin, что и UI (`VITE_API_BASE` не задан).
- Для разработки можно указать `VITE_API_BASE=http://127.0.0.1:8765` в `.env.local`.
- Оболочка передаёт токен во фрагменте URL: `#token=...`. UI сохраняет его в `sessionStorage`
  и сразу очищает адресную строку через `history.replaceState`.
- Запросы к API отправляют `Authorization: Bearer <token>`. Ответ `401` показывает экран
  «Нет доступа к ядру» и подсказку перезапустить приложение.
- `/healthz` опрашивается каждые 5 секунд для индикатора состояния ядра.

## Структура

```
src/
  api.ts               типы и клиент локального API
  theme.ts             светлая/тёмная темы Fluent, бренд #0F6CBD
  App.tsx              оболочка, навигация, тема, auth/health
  components/          Sidebar, Dropzone, PageHeader, состояния и пр.
  hooks/               useAppTheme, useCoreStatus
  pages/               Очередь, База знаний, Модели, Настройки
  styles.css           Mica-фон, layout, карточки, адаптивность
```

## Границы M0

- Кнопка «Добавить файлы» и drag&drop только собирают имена файлов в окне.
  `POST /api/v1/jobs` не вызывается.
- «Установить» (модели) и «Импортировать документы» (база знаний) отключены до M1/M4.
- Настройки читаются и сохраняются полностью через `GET`/`PUT /api/v1/settings`.
- Тема применяется сразу, но сохраняется на сервере только по кнопке «Сохранить».

## Проверка

```powershell
npm.cmd run typecheck
npm.cmd run build
```

Целевое окно — 1200×800, интерфейс остаётся рабочим при 960×640.
