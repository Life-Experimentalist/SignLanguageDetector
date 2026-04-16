# CFlair Counter Integration

This repository uses [CFlair-Counter](https://github.com/Life-Experimentalist/CFlair-Counter?tab=readme-ov-file#cflair-counter) to collect anonymous view stats and expose a badge for project usage.

## Base URL

`https://counter.vkrishna04.me`

## Quick API Reference

- `POST /api/views/:projectName` - Increment view count for a project.
- `GET /api/views/:projectName` - Fetch counts for a project.
- `GET /api/views/:projectName/badge` - Render an SVG badge.
- `GET /api/stats` - Global statistics across projects.
- `POST /api/admin/stats` - Admin statistics endpoint.
- `GET /health` - Health check.

## Integration Steps

1. Add the badge or counter call in your project.
2. Keep the project name stable, for example `sign-language-detector`.
3. Use the POST endpoint on page load or other meaningful events.
4. Keep telemetry non-blocking so it never affects the app.

## Example Request

```js
fetch("https://counter.vkrishna04.me/api/views/sign-language-detector", {
  method: "POST",
});
```

## Badge Example

```markdown
![Project Views](https://counter.vkrishna04.me/api/views/sign-language-detector/badge?style=flat-square&color=brightgreen&label=views)
```

## Environment Variables

- `DISABLE_ANONYMOUS_TELEMETRY` - Preferred toggle to disable telemetry.
- `DISABLE_ANONYMOUS_TELMETRY` - Legacy compatibility toggle.
- `TELEMETRY_COUNTER_BASE_URL` - Counter base URL.
- `TELEMETRY_PROJECT_NAME` - Project key sent to the counter.

## Security Notes

- Keep admin credentials out of source control.
- Telemetry should remain anonymous.
- UI behavior must not depend on telemetry success.
