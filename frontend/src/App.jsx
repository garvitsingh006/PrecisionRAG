import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Nav from './components/Nav'
import Landing from './pages/Landing'
import Upload from './pages/Upload'
import Query from './pages/Query'
import Config from './pages/Config'
import Resume from './pages/Resume'

export default function App() {
  return (
    <BrowserRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/query" element={<Query />} />
        <Route path="/config" element={<Config />} />
        <Route path="/resume" element={<Resume />} />
      </Routes>
    </BrowserRouter>
  )
}
