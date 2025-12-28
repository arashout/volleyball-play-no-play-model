  (async function extractBallTimeActions() {
    const container = document.querySelector('div[style*="overflow: auto"]');
    if (!container) { console.error('Container not found'); return; }

    const actions = new Map();
    const totalHeight = parseInt(container.querySelector('ul').style.height);
    const viewHeight = container.clientHeight;

    for (let scrollPos = 0; scrollPos <= totalHeight; scrollPos += viewHeight * 0.8) {
      container.scrollTop = scrollPos;
      await new Promise(r => setTimeout(r, 150));

      container.querySelectorAll('li').forEach(li => {
        const idx = li.dataset.index;
        if (actions.has(idx)) return;

        const timeSpan = li.querySelector('.item-content span');
        const actionBold = li.querySelectorAll('b');
        const rallyHeader = li.querySelector('.py-0\\.5');

        if (rallyHeader) {
          const rallyText = rallyHeader.textContent.trim();
          actions.set(idx, { type: 'rally', text: rallyText });
        } else if (timeSpan && actionBold.length > 0) {
          const time = timeSpan.textContent.trim();
          const actionType = actionBold[0]?.textContent || '';
          const result = actionBold[1]?.textContent || '';
          actions.set(idx, { type: 'action', time, actionType, result });
        }
      });
    }

    const sorted = [...actions.entries()]
      .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
      .map(([_, v]) => v);

    console.log('\n=== EXTRACTED ACTIONS ===\n');
    sorted.forEach(item => {
      if (item.type === 'rally') {
        console.log(`\n${item.text}`);
      } else {
        console.log(`${item.time} - ${item.actionType} (${item.result})`);
      }
    });

    const csvData = sorted
      .filter(i => i.type === 'action')
      .map(i => `${i.time},${i.actionType},${i.result}`)
      .join('\n');
    console.log('\n=== CSV FORMAT ===\ntime,action,result\n' + csvData);

    return sorted;
  })();