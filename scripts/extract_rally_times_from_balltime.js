(async function extractRallyTimings() {
    const container = document.querySelector('div[style*="overflow: auto"]');
    if (!container) { console.error('Container not found'); return; }

    const rallies = new Map();
    const totalHeight = parseInt(container.querySelector('ul').style.height);
    const viewHeight = container.clientHeight;

    for (let scrollPos = 0; scrollPos <= totalHeight; scrollPos += viewHeight * 0.8) {
      container.scrollTop = scrollPos;
      await new Promise(r => setTimeout(r, 150));

      container.querySelectorAll('li').forEach(li => {
        const idx = li.dataset.index;
        if (rallies.has(idx)) return;

        const timeDiv = li.querySelector('.item-content > div');
        const rallyDiv = li.querySelector('.line-clamp-2');

        if (timeDiv && rallyDiv) {
          const time = timeDiv.textContent.trim();
          const rally = rallyDiv.textContent.trim();
          rallies.set(idx, { time, rally });
        }
      });
    }

    const sorted = [...rallies.entries()]
      .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
      .map(([_, v]) => v);

    console.log('\n=== RALLY TIMINGS ===\n');
    sorted.forEach(r => console.log(`${r.rally}: ${r.time}`));

    const csv = 'rally,start,end\n' + sorted.map(r => {
      const [start, end] = r.time.split(' - ');
      return `${r.rally},${start},${end}`;
    }).join('\n');
    console.log('\n=== CSV ===\n' + csv);

    return sorted;
  })();