import { Button } from '@fluentui/react-components';
import { ArrowClockwiseRegular, LockClosedRegular } from '@fluentui/react-icons';

interface AuthErrorPanelProps {
  onRetry: () => void;
}

/** Shown when the core rejects the session token with HTTP 401. */
export function AuthErrorPanel({ onRetry }: AuthErrorPanelProps) {
  return (
    <div className="auth-screen">
      <section className="auth-panel surface" role="alert" aria-labelledby="auth-title">
        <span className="auth-icon" aria-hidden="true">
          <LockClosedRegular fontSize={26} />
        </span>
        <h1 id="auth-title" className="auth-title">
          Нет доступа к ядру
        </h1>
        <p className="auth-text">
          Оболочка не передала актуальный токен доступа. Перезапустите AI Recogniser — токен
          подставляется автоматически при открытии окна.
        </p>
        <p className="auth-hint">
          Если окно открыто вручную в браузере, запустите приложение через AI Recogniser.exe.
        </p>
        <Button
          appearance="primary"
          icon={<ArrowClockwiseRegular />}
          onClick={onRetry}
        >
          Проверить снова
        </Button>
      </section>
    </div>
  );
}
