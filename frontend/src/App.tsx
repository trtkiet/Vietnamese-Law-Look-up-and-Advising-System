import { useMemo, useState } from 'react'
import { Navigate, NavLink, Route, Routes } from 'react-router-dom'
import { Badge, Button, Input, Panel, PrimaryButton } from './ui'

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2 text-lg font-semibold">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-slate-900 text-white">VA</span>
            <span>Vietnamese Law Assistant</span>
          </div>
          <nav className="flex items-center gap-4 text-sm font-medium text-slate-600">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `rounded-md px-3 py-2 transition ${isActive ? 'bg-slate-900 text-white' : 'hover:bg-slate-100'}`
              }
            >
              Chatbot
            </NavLink>
            <NavLink
              to="/lookup"
              className={({ isActive }) =>
                `rounded-md px-3 py-2 transition ${isActive ? 'bg-slate-900 text-white' : 'hover:bg-slate-100'}`
              }
            >
              Lookup
            </NavLink>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/lookup" element={<LookupPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}

function ChatPage() {
  const initialMessages = useMemo(
    () => [
      { role: 'ai', text: 'Chào bạn, tôi có thể giúp trả lời câu hỏi pháp lý và trích dẫn điều khoản liên quan.' },
      { role: 'user', text: 'Quy định về thử việc trong Bộ luật Lao động 2019 thế nào?' },
      {
        role: 'ai',
        text:
          'Thời gian thử việc tùy thuộc vào vị trí, tối đa 180 ngày với quản lý, 60 ngày với trình độ cao đẳng trở lên, 30 ngày với trung cấp, 6 ngày với công việc khác.',
      },
    ],
    [],
  )

  const sources = [
    { title: 'Bộ luật Lao động 2019', ref: 'Điều 25', snippet: 'Thời gian thử việc tùy thuộc vào tính chất, mức độ phức tạp của công việc...' },
    { title: 'Nghị định 145/2020/NĐ-CP', ref: 'Điều 7', snippet: 'Hướng dẫn chi tiết về thử việc và hợp đồng lao động.' },
  ]

  const [composer, setComposer] = useState('')

  return (
    <div className="grid grid-cols-5 gap-6">
      <Panel className="col-span-3 p-6">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Conversation</p>
            <h2 className="text-lg font-semibold text-slate-900">Assistant</h2>
          </div>
          <Badge>Mock data</Badge>
        </div>

        <div className="flex flex-col gap-4 pb-4">
          {initialMessages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
                  msg.role === 'user' ? 'bg-slate-900 text-white' : 'bg-slate-50 text-slate-900'
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-2 flex flex-col gap-3 border-t border-slate-200 pt-4">
          <div className="flex gap-2">
            <Input
              value={composer}
              onChange={(e) => setComposer(e.target.value)}
              placeholder="Nhập câu hỏi pháp lý của bạn..."
            />
            <PrimaryButton type="button" disabled>
              Send
            </PrimaryButton>
          </div>
          <p className="text-xs text-slate-500">Sending is stubbed for now; responses are mocked.</p>
        </div>
      </Panel>

      <div className="col-span-2 flex flex-col gap-4">
        <Panel className="p-5">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500">Document</p>
              <h3 className="text-base font-semibold text-slate-900">Bộ luật Lao động 2019</h3>
            </div>
            <Button type="button">Open full text</Button>
          </div>
          <div className="space-y-2 text-sm text-slate-700">
            <p>
              Bộ luật này quy định tiêu chuẩn lao động; quyền, nghĩa vụ, trách nhiệm của người lao động, người sử dụng lao động, tổ chức đại diện người lao động...
            </p>
            <p>
              Thời gian thử việc được nêu tại Điều 25 với các mốc 180/60/30/6 ngày tùy vị trí. Sau khi kết thúc thử việc, phải thông báo kết quả thử việc.
            </p>
          </div>
        </Panel>

        <Panel className="p-5">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-xs uppercase tracking-wide text-slate-500">Sources</p>
            <Badge>2 references</Badge>
          </div>
          <div className="space-y-3">
            {sources.map((source) => (
              <div key={source.ref} className="rounded-md border border-slate-200 bg-slate-50 p-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-900">{source.title}</span>
                  <Badge className="bg-white">{source.ref}</Badge>
                </div>
                <p className="mt-1 text-slate-700">{source.snippet}</p>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  )
}

function LookupPage() {
  const [query, setQuery] = useState('thử việc')
  const [selectedId, setSelectedId] = useState('boll-25')

  const results = [
    {
      id: 'boll-25',
      title: 'Bộ luật Lao động 2019',
      type: 'Luật',
      ref: 'Điều 25',
      date: '2019',
      snippet: 'Thời gian thử việc tùy thuộc vào tính chất công việc và trình độ chuyên môn của người lao động...',
    },
    {
      id: 'nd-145-7',
      title: 'Nghị định 145/2020/NĐ-CP',
      type: 'Nghị định',
      ref: 'Điều 7',
      date: '2020',
      snippet: 'Hướng dẫn chi tiết về thử việc, hợp đồng lao động và trách nhiệm thông báo kết quả thử việc...',
    },
    {
      id: 'qđ-999',
      title: 'Quyết định 999/QĐ-BLĐTBXH',
      type: 'Quyết định',
      ref: 'Điều 3',
      date: '2021',
      snippet: 'Quy định biểu mẫu thông báo kết quả thử việc trong doanh nghiệp...',
    },
  ]

  const active = results.find((item) => item.id === selectedId) ?? results[0]

  return (
    <div className="grid grid-cols-5 gap-6">
      <div className="col-span-2">
        <Panel className="p-5">
          <div className="flex items-center gap-2">
            <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Tìm kiếm văn bản pháp luật" />
            <Button type="button">Search</Button>
          </div>
          <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-600">
            <Badge>Luật</Badge>
            <Badge>Nghị định</Badge>
            <Badge>Pháp lệnh</Badge>
            <Badge>Quyết định</Badge>
          </div>
          <div className="mt-5 space-y-3">
            {results.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setSelectedId(item.id)}
                className={`w-full rounded-md border px-3 py-3 text-left text-sm transition ${
                  item.id === active.id
                    ? 'border-slate-300 bg-slate-100 shadow-sm'
                    : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-900">{item.title}</span>
                  <Badge>{item.type}</Badge>
                </div>
                <div className="mt-1 text-xs text-slate-600">{item.ref} • {item.date}</div>
                <p className="mt-2 line-clamp-2 text-slate-700">{item.snippet}</p>
              </button>
            ))}
          </div>
        </Panel>
      </div>

      <Panel className="col-span-3 p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-500">Preview</p>
            <h2 className="text-lg font-semibold text-slate-900">{active.title}</h2>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <Badge>{active.type}</Badge>
            <Badge>{active.ref}</Badge>
            <Badge>{active.date}</Badge>
          </div>
        </div>
        <div className="mt-4 space-y-3 text-sm text-slate-800">
          <p>
            {active.snippet} Đây là dữ liệu giả lập; bạn có thể nối API tìm kiếm để thay thế. Nội dung đầy đủ văn bản sẽ hiển thị ở khu vực này khi tích hợp.
          </p>
          <p>
            Liên kết trích dẫn và tải xuống (nếu có) cũng có thể đặt tại đây để người dùng mở văn bản gốc.
          </p>
        </div>
      </Panel>
    </div>
  )
}