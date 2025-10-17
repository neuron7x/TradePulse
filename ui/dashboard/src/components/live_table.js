import { escapeHtml } from '../core/formatters.js';

const DEFAULT_PAGE_SIZE = 10;
const SORT_DIRECTIONS = new Set(['asc', 'desc']);

function normaliseDirection(direction = 'desc') {
  const lower = String(direction).toLowerCase();
  return SORT_DIRECTIONS.has(lower) ? lower : 'desc';
}

function clampPage(page, pageCount) {
  if (!Number.isFinite(page) || page < 1) {
    return 1;
  }
  if (page > pageCount) {
    return pageCount;
  }
  return page;
}

export class LiveTable {
  constructor({ columns = [], rows = [], sortBy, sortDirection = 'desc', pageSize = DEFAULT_PAGE_SIZE } = {}) {
    if (!Array.isArray(columns) || columns.length === 0) {
      throw new Error('LiveTable requires at least one column definition');
    }
    this.columns = columns.map((column) => ({
      id: column.id,
      label: column.label,
      accessor: typeof column.accessor === 'function' ? column.accessor : (row) => row[column.id],
      formatter: column.formatter,
      align: column.align || 'left',
      sortValue: column.sortValue,
    }));
    this.rows = Array.isArray(rows) ? rows.slice() : [];
    this.sortBy = sortBy || this.columns[0].id;
    this.sortDirection = normaliseDirection(sortDirection);
    this.pageSize = Number.isFinite(pageSize) && pageSize > 0 ? Math.floor(pageSize) : DEFAULT_PAGE_SIZE;
    this.page = 1;
  }

  setRows(rows = []) {
    this.rows = Array.isArray(rows) ? rows.slice() : [];
    return this;
  }

  setSort(columnId, direction = this.sortDirection) {
    if (!this.columns.find((column) => column.id === columnId)) {
      throw new Error(`Unknown column: ${columnId}`);
    }
    this.sortBy = columnId;
    this.sortDirection = normaliseDirection(direction);
    return this;
  }

  setPage(page) {
    this.page = Number.isFinite(page) ? Math.max(1, Math.floor(page)) : 1;
    return this;
  }

  setPageSize(size) {
    if (!Number.isFinite(size) || size <= 0) {
      throw new Error('Page size must be a positive number');
    }
    this.pageSize = Math.floor(size);
    return this;
  }

  getSortedRows() {
    const column = this.columns.find((col) => col.id === this.sortBy) || this.columns[0];
    const rows = this.rows.slice();
    const directionMultiplier = this.sortDirection === 'asc' ? 1 : -1;
    const accessor = column.accessor;
    const sortValue = column.sortValue;
    return rows.sort((a, b) => {
      const aValue = sortValue ? sortValue(a) : accessor(a);
      const bValue = sortValue ? sortValue(b) : accessor(b);
      if (aValue === bValue) {
        return 0;
      }
      if (aValue === undefined || aValue === null) {
        return 1 * directionMultiplier;
      }
      if (bValue === undefined || bValue === null) {
        return -1 * directionMultiplier;
      }
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return (aValue - bValue) * directionMultiplier;
      }
      return String(aValue).localeCompare(String(bValue)) * directionMultiplier;
    });
  }

  renderRow(row) {
    const cells = this.columns
      .map((column) => {
        const rawValue = column.accessor(row);
        const display = column.formatter ? column.formatter(rawValue, row) : escapeHtml(rawValue ?? '—');
        return `<td class="tp-live-table__cell tp-live-table__cell--${column.align}">${display}</td>`;
      })
      .join('');
    return `<tr class="tp-live-table__row">${cells}</tr>`;
  }

  render(page = this.page) {
    const sortedRows = this.getSortedRows();
    const pageCount = Math.max(Math.ceil(sortedRows.length / this.pageSize), 1);
    const currentPage = clampPage(page, pageCount);
    const start = (currentPage - 1) * this.pageSize;
    const end = start + this.pageSize;
    const pageRows = sortedRows.slice(start, end);

    const body = pageRows.length
      ? pageRows.map((row) => this.renderRow(row)).join('')
      : `<tr class="tp-live-table__row tp-live-table__row--empty"><td class="tp-live-table__cell" colspan="${this.columns.length}">No data available.</td></tr>`;

    const header = this.columns
      .map((column) => {
        const isActive = column.id === this.sortBy;
        const indicator = isActive ? `<span class="tp-live-table__sort">${this.sortDirection === 'asc' ? '▲' : '▼'}</span>` : '';
        return `<th class="tp-live-table__header tp-live-table__cell--${column.align}" scope="col" data-column="${escapeHtml(column.id)}">${escapeHtml(column.label)}${indicator}</th>`;
      })
      .join('');

    const html = `
      <div class="tp-live-table">
        <table class="tp-live-table__table">
          <thead class="tp-live-table__head">
            <tr class="tp-live-table__row">${header}</tr>
          </thead>
          <tbody class="tp-live-table__body">${body}</tbody>
        </table>
        <footer class="tp-live-table__footer">
          <span class="tp-live-table__footer-item">Page ${currentPage} of ${pageCount}</span>
          <span class="tp-live-table__footer-item">Rows ${sortedRows.length}</span>
        </footer>
      </div>
    `;

    return {
      html,
      page: currentPage,
      pageCount,
      totalRows: sortedRows.length,
      pageSize: this.pageSize,
    };
  }
}

export function createLiveTable(config) {
  return new LiveTable(config);
}
