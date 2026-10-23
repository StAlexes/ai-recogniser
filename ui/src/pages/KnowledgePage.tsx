import { Badge, Button } from '@fluentui/react-components';
import {
  ArrowUploadRegular,
  BookOpenRegular,
  DatabaseRegular,
  KeyRegular,
  SearchRegular,
} from '@fluentui/react-icons';
import { DisabledAction } from '../components/DisabledAction';
import { PageHeader } from '../components/PageHeader';

export function KnowledgePage() {
  return (
    <>
      <PageHeader
        title="База знаний"
        subtitle="Документы и термины, на которые опираются саммари"
        badge={
          <Badge appearance="tint" color="informative">
            M4
          </Badge>
        }
        actions={
          <DisabledAction action="Импортировать документы" tooltip="Появится в M4">
            <Button icon={<ArrowUploadRegular />} disabled>
              Импортировать документы
            </Button>
          </DisabledAction>
        }
      />

      <section className="kb-hero surface" aria-labelledby="kb-hero-title">
        <span className="kb-hero-icon" aria-hidden="true">
          <BookOpenRegular fontSize={26} />
        </span>
        <div className="kb-hero-body">
          <h2 id="kb-hero-title" className="kb-hero-title">
            Контекст для точных ответов
          </h2>
          <p className="kb-hero-text">
            RAG находит нужные фрагменты в ваших документах, добавляет их в промпт и заставляет
            модель ссылаться на источники. Без облака и без отправки файлов наружу.
          </p>
          <p className="kb-hero-note">
            Импорт pdf, docx, txt и md, чанкинг, эмбеддинги и гибридный поиск появятся в M4.
          </p>
        </div>
      </section>

      <div className="kb-grid">
        <article className="feature-card surface">
          <span className="feature-icon" aria-hidden="true">
            <DatabaseRegular fontSize={20} />
          </span>
          <h3 className="feature-title">Индексация</h3>
          <p className="feature-text">
            Чанкинг с перекрытием, эмбеддинги multilingual-e5 и хранение в sqlite-vec с запасным
            поиском.
          </p>
        </article>

        <article className="feature-card surface">
          <span className="feature-icon" aria-hidden="true">
            <SearchRegular fontSize={20} />
          </span>
          <h3 className="feature-title">Ответы с источниками</h3>
          <p className="feature-text">
            Гибридный поиск: вектор и BM25. В саммари и Q&amp;A попадают ссылки на документы и
            страницы.
          </p>
        </article>

        <article className="feature-card surface">
          <span className="feature-icon" aria-hidden="true">
            <KeyRegular fontSize={20} />
          </span>
          <h3 className="feature-title">Глоссарий</h3>
          <p className="feature-text">
            Пары «как слышится → как правильно»: «битрикс» → «Битрикс24», термины и пояснения для
            ASR и LLM.
          </p>
        </article>
      </div>
    </>
  );
}
